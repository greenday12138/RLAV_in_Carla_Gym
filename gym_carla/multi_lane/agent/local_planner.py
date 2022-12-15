import carla
import copy
from collections import deque
from shapely.geometry import Polygon
from gym_carla.multi_lane.agent.global_planner import RoadOption
from gym_carla.multi_lane.settings import ROADS, STRAIGHT, CURVE, JUNCTION, DOUBLE_DIRECTION, DISTURB_ROADS
from gym_carla.multi_lane.util.misc import get_lane_center, get_speed, vector, compute_magnitude_angle, \
    is_within_distance_ahead, is_within_distance_rear, draw_waypoints, compute_distance, is_within_distance, test_waypoint


class LocalPlanner:
    def __init__(self, vehicle, opt_dict=None):
        if opt_dict is None:
            opt_dict = {'sampling_resolution': 4.0,
                        'buffer_size': 10,
                        'vehicle_proximity': 50
                        }
        """
            temporarily used to get front waypoints and vehicle
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._sampling_radius = opt_dict['sampling_resolution']
        self._base_min_distance = 3.0  # This value is tricky

        self._target_waypoint = None
        self._buffer_size = opt_dict['buffer_size']
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._waypoints_queue = deque(maxlen=600)
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._target_road_option = RoadOption.LANEFOLLOW
        self._stop_waypoint_creation = False

        self._last_traffic_light = None
        self._proximity_threshold = opt_dict['vehicle_proximity']

        self._waypoints_queue.append((self._current_waypoint, RoadOption.LANEFOLLOW))
        # self._waypoints_queue.append( (self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        # self._compute_next_waypoints(k=200)

    def run_step(self):
        waypoints = self._get_waypoints()
        red_light, vehicle_front = self._get_hazard()
        # red_light = False
        return waypoints, red_light, vehicle_front

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self._waypoint_buffer) > steps:
            return self._waypoint_buffer[steps]
        else:
            try:
                wpt, direction = self._waypoint_buffer[-1]
                return wpt, direction
            except IndexError as i:
                return None, RoadOption.VOID

    def set_sampling_redius(self, sampling_resolution):
        self._sampling_radius = sampling_resolution

    def set_min_distance(self, min_distance):
        self._min_distance = min_distance

    def set_global_plan(self, current_plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs
        The 'clean_queue` parameter erases the previous plan if True, otherwise, it adds it to the old one
        The 'stop_waypoint_creation' flag stops the automatic creation of random waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :param stop_waypoint_creation: bool
        :param clean_queue: bool
        :return:
        """
        if clean_queue:
            self._waypoints_queue.clear()

        # Remake the waypoints queue if the new plan has a higher length than the queue
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue

        for elem in current_plan:
            self._waypoints_queue.append((elem, RoadOption.LANEFOLLOW))

        self._stop_waypoint_creation = stop_waypoint_creation

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                road_options_list = self._retrieve_options(
                    next_waypoints, last_waypoint)

                # # random choice between the possible options
                # road_option = road_options_list[1]
                # #road_option = random.choice(road_options_list)
                # next_waypoint = next_waypoints[road_options_list.index(road_option)]

                idx = None
                for i, wp in enumerate(next_waypoints):
                    if wp.road_id in ROADS:
                        next_waypoint = wp
                        idx = i
                road_option = road_options_list[idx]

            self._waypoints_queue.append((next_waypoint, road_option))

    def _get_waypoints(self):
        """Get the next waypoint list according to ego vehicle's current location"""
        lane_center = get_lane_center(self._map, self._vehicle.get_location())
        _waypoints_queue = deque(maxlen=600)
        _waypoints_queue.append(lane_center)
        available_entries = _waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, self._buffer_size)

        for _ in range(k):
            last_waypoint = _waypoints_queue[-1]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                #road_option = RoadOption.LANEFOLLOW
            else:
                # road_options_list = self._retrieve_options(
                #     next_waypoints, last_waypoint)

                idx = None
                for i, wp in enumerate(next_waypoints):
                    if wp.road_id in ROADS:
                        next_waypoint = wp
                        idx = i
                #road_option = road_options_list[idx]

            _waypoints_queue.append(next_waypoint)
        # delete an element from the left
        _waypoints_queue.popleft()
        return _waypoints_queue

    def get_waypoint_one_lane(self, buffer_size, waypoint=None):
        _waypoints_queue = deque(maxlen=600)
        if waypoint is not None:
            _waypoints_queue.append(waypoint)
            available_entries = _waypoints_queue.maxlen - len(self._waypoints_queue)
            k = min(available_entries, buffer_size)
            for _ in range(k):
                last_waypoint = _waypoints_queue[-1]
                next_waypoints = list(last_waypoint.next(self._sampling_radius))

                if len(next_waypoints) == 0:
                    break
                elif len(next_waypoints) == 1:
                    # only one option available ==> lanefollowing
                    next_waypoint = next_waypoints[0]
                    # road_option = RoadOption.LANEFOLLOW
                else:
                    # road_options_list = self._retrieve_options(
                    #     next_waypoints, last_waypoint)

                    idx = None
                    for i, wp in enumerate(next_waypoints):
                        if wp.road_id in ROADS:
                            next_waypoint = wp
                            idx = i
                    # road_option = road_options_list[idx]

                _waypoints_queue.append(next_waypoint)
            # delete an element from the left
            _waypoints_queue.popleft()
        return _waypoints_queue

    def get_rear_waypoint_one_lane(self, buffer_size, waypoint=None):
        _waypoints_queue = deque(maxlen=600)
        if waypoint is not None:
            _waypoints_queue.append(waypoint)
            available_entries = _waypoints_queue.maxlen - len(self._waypoints_queue)
            k = min(available_entries, buffer_size)
            for _ in range(k):
                last_waypoint = _waypoints_queue[-1]
                previous_waypoints = list(last_waypoint.previous(self._sampling_radius))

                if len(previous_waypoints) == 0:
                    break
                elif len(previous_waypoints) == 1:
                    # only one option available ==> lanefollowing
                    previous_waypoint = previous_waypoints[0]
                    # road_option = RoadOption.LANEFOLLOW
                else:
                    # road_options_list = self._retrieve_options(
                    #     next_waypoints, last_waypoint)
                    idx = None
                    for i, wp in enumerate(previous_waypoints):
                        if wp.road_id in ROADS:
                            previous_waypoint = wp
                            idx = i
                    # road_option = road_options_list[idx]

                _waypoints_queue.append(previous_waypoint)
            # delete an element from the left
            _waypoints_queue.popleft()
        return _waypoints_queue

    def _get_waypoints_multilane(self):
        """
        :return: front waypoints (self._buffer_size) in three lanes
        """
        lane_center = get_lane_center(self._map, self._vehicle.get_location())
        lane_id = lane_center.lane_id
        left = None
        center = lane_center
        right = None
        if lane_id == -1:
            right = center.get_right_lane()
            right_right = right.get_right_lane()
            return [self.get_waypoint_one_lane(self._buffer_size, left),
               self.get_waypoint_one_lane(self._buffer_size, center),
               self.get_waypoint_one_lane(self._buffer_size, right),
               self.get_rear_waypoint_one_lane(self._buffer_size, left),
               self.get_rear_waypoint_one_lane(self._buffer_size, center),
               self.get_rear_waypoint_one_lane(self._buffer_size, right)], [self.get_waypoint_one_lane(self._buffer_size, center),
             self.get_waypoint_one_lane(self._buffer_size, right),
             self.get_waypoint_one_lane(self._buffer_size, right_right),
             self.get_rear_waypoint_one_lane(self._buffer_size, center),
             self.get_rear_waypoint_one_lane(self._buffer_size, right),
             self.get_rear_waypoint_one_lane(self._buffer_size, right_right)]
        elif lane_id == -2:
            left = center.get_left_lane()
            right = center.get_right_lane()
            return [self.get_waypoint_one_lane(self._buffer_size, left),
               self.get_waypoint_one_lane(self._buffer_size, center),
               self.get_waypoint_one_lane(self._buffer_size, right),
               self.get_rear_waypoint_one_lane(self._buffer_size, left),
               self.get_rear_waypoint_one_lane(self._buffer_size, center),
               self.get_rear_waypoint_one_lane(self._buffer_size, right)], [self.get_waypoint_one_lane(self._buffer_size, left),
             self.get_waypoint_one_lane(self._buffer_size, center),
             self.get_waypoint_one_lane(self._buffer_size, right),
             self.get_rear_waypoint_one_lane(self._buffer_size, left),
             self.get_rear_waypoint_one_lane(self._buffer_size, center),
             self.get_rear_waypoint_one_lane(self._buffer_size, right)]
        elif lane_id == -3:
            left = center.get_left_lane()
            left_left = left.get_left_lane()
            return [self.get_waypoint_one_lane(self._buffer_size, left),
               self.get_waypoint_one_lane(self._buffer_size, center),
               self.get_waypoint_one_lane(self._buffer_size, right),
               self.get_rear_waypoint_one_lane(self._buffer_size, left),
               self.get_rear_waypoint_one_lane(self._buffer_size, center),
               self.get_rear_waypoint_one_lane(self._buffer_size, right)], [self.get_waypoint_one_lane(self._buffer_size, center),
             self.get_waypoint_one_lane(self._buffer_size, left),
             self.get_waypoint_one_lane(self._buffer_size, left_left),
             self.get_rear_waypoint_one_lane(self._buffer_size, center),
             self.get_rear_waypoint_one_lane(self._buffer_size, left),
             self.get_rear_waypoint_one_lane(self._buffer_size, left_left)]
        else:
            # just to avoid error, we will not use the return
            left = lane_center.get_left_lane()
            center = lane_center
            right = lane_center.get_right_lane()
            return [self.get_waypoint_one_lane(self._buffer_size, left),
                    self.get_waypoint_one_lane(self._buffer_size, center),
                    self.get_waypoint_one_lane(self._buffer_size, right),
                    self.get_rear_waypoint_one_lane(self._buffer_size, left),
                    self.get_rear_waypoint_one_lane(self._buffer_size, center),
                    self.get_rear_waypoint_one_lane(self._buffer_size, right)], [
                       self.get_waypoint_one_lane(self._buffer_size, left),
                       self.get_waypoint_one_lane(self._buffer_size, center),
                       self.get_waypoint_one_lane(self._buffer_size, right),
                       self.get_rear_waypoint_one_lane(self._buffer_size, left),
                       self.get_rear_waypoint_one_lane(self._buffer_size, center),
                       self.get_rear_waypoint_one_lane(self._buffer_size, right)]


    # def _get_waypoints(self):
    #     """
    #     Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
    #     follow the waypoints trajectory.

    #     :param debug: boolean flag to activate waypoints debugging
    #     :return:
    #     """

    #     # not enough waypoints in the horizon? => add more!
    #     if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5) and not self._stop_waypoint_creation:
    #         self._compute_next_waypoints(self._buffer_size * 2)

    #     #   Buffering the waypoints
    #     while len(self._waypoint_buffer) < self._buffer_size:
    #         if self._waypoints_queue:
    #             self._waypoint_buffer.append(
    #                 self._waypoints_queue.popleft())
    #         else:
    #             break

    #     waypoints = []

    #     for i, (waypoint, _) in enumerate(self._waypoint_buffer):
    #         waypoints.append(waypoint)
    #         # waypoints.append([waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.rotation.yaw])

    #     # current vehicle waypoint
    #     self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
    #     # target waypoint
    #     self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]

    #     # purge the queue of obsolete waypoints
    #     # vehicle_transform = self._vehicle.get_transform()
    #     # max_index = -1

    #     # for i, (waypoint, _) in enumerate(self._waypoint_buffer):
    #     #     if distance_vehicle(waypoint, vehicle_transform) < self._min_distance:
    #     #         max_index = i
    #     # if max_index >= 0:
    #     #     for i in range(max_index - 1):
    #     #         self._waypoint_buffer.popleft()

    #     veh_location = self._vehicle.get_location()
    #     veh_speed = get_speed(self._vehicle, False)
    #     settings = self._world.get_settings()
    #     if settings.synchronous_mode:
    #         self._min_distance = self._base_min_distance + settings.fixed_delta_seconds * veh_speed
    #     else:
    #         self._min_distance = self._base_min_distance + 0.5 * veh_speed
    #     num_waypoint_removed = 0
    #     for waypoint, _ in self._waypoint_buffer:

    #         if len(self._waypoints_queue) - num_waypoint_removed == 1:
    #             min_distance = 1  # Don't remove the last waypoint until very close by
    #         else:
    #             min_distance = self._min_distance

    #         if veh_location.distance(waypoint.transform.location) < min_distance:
    #             num_waypoint_removed += 1
    #         else:
    #             break

    #     if num_waypoint_removed > 0:
    #         for _ in range(num_waypoint_removed):
    #             self._waypoint_buffer.popleft()

    #             # lane_center=get_lane_center(self._map,self._vehicle.get_location())
    #     # print(lane_center.road_id,lane_center.lane_id,lane_center.s,sep='\t',end='\n\n')
    #     # for wp,_ in self._waypoint_buffer:
    #     #     print(wp.road_id,wp.lane_id,wp.s,wp.transform.location.distance(lane_center.transform.location),sep='\t')

    #     return waypoints

    def _get_hazard(self):
        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle = self._get_front_vehicle(vehicle_list)

        # check for the state of the traffic lights
        light_state = self._is_light_red_us_style(lights_list)

        return light_state, vehicle

    def _get_traffic_light(self):
        """
        TODO: detected distance of traffic light
        :return:
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        light_state = self._is_light_red_us_style(lights_list)
        return light_state

    def _get_front_rear_inlane_vehicle(self):
        """

        :return: the front vehicle in the same lane
        the rear vehicle in the same lane
        detected surrounding vehicles in three lanes
        """
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")

        # check possible obstacles
        front_vehicle = self._get_front_vehicle(vehicle_list, 0)
        rear_vehicle = self._get_rear_vehicle(vehicle_list, 0)
        in_lane_vehicles = self._get_in_lane_vehicles(vehicle_list)

        return front_vehicle, rear_vehicle, in_lane_vehicles

    def _get_in_lane_vehicles(self, vehicle_list):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: detected surrounding vehicles in three lanes
        """
        front_left_vehicle = self._get_front_vehicle(vehicle_list, -1)
        front_vehicle = self._get_front_vehicle(vehicle_list, 0)
        front_right_vehicle = self._get_front_vehicle(vehicle_list, 1)
        rear_left_vehicle = self._get_rear_vehicle(vehicle_list, -1)
        rear_vehicle = self._get_rear_vehicle(vehicle_list, 0)
        rear_right_vehicle = self._get_rear_vehicle(vehicle_list, 1)
        # if front_vehicle != None:
        #     print('front_vehicle.location: ', front_vehicle.get_location())
        # if rear_vehicle != None:
        #     print('rear_vehicle.location: ', rear_vehicle.get_location())
        return [front_left_vehicle, front_vehicle, front_right_vehicle, rear_left_vehicle, rear_vehicle, rear_right_vehicle]

    def _get_rear_vehicle(self, vehicle_list, direction=0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        behind our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
        vehicles, which center is actually on a different lane but their
        extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return:
            - the first rear vehicle
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_transform = self._vehicle.get_transform()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        ego_vehicle_lane_center = get_lane_center(self._map, ego_vehicle_location)
        min_distance = self._proximity_threshold
        vehicle_rear = None
        lane_id = ego_vehicle_lane_center.lane_id - direction
        if lane_id != -1 and lane_id != -2 and lane_id != -3:
            return vehicle_rear

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            # check whether in the same road
            target_lane_center = get_lane_center(self._map, target_vehicle.get_location())
            if target_lane_center.transform.location.distance(target_vehicle.get_location()) > target_lane_center.lane_width / 2 + 0.1:
                continue
            if not test_waypoint(target_vehicle_waypoint):
                continue
            # check whether in the specific lane
            if target_vehicle_waypoint.lane_id != lane_id:
                continue
            # if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
            #         target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            #     continue

            loc = target_vehicle.get_location()
            if is_within_distance_rear(loc, ego_vehicle_location, ego_vehicle_transform, self._proximity_threshold):
                if ego_vehicle_location.distance(loc) < min_distance:
                    # Return the most close vehicel in front of ego vehicle
                    vehicle_rear = target_vehicle
                    min_distance = ego_vehicle_location.distance(loc)

        return vehicle_rear

    def _get_front_vehicle(self, vehicle_list, direction=0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
        vehicles, which center is actually on a different lane but their
        extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return:
            - vehicle is the blocker object itself
            - the front vehicle
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_transform = self._vehicle.get_transform()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        ego_vehicle_lane_center = get_lane_center(self._map, ego_vehicle_location)
        min_distance = self._proximity_threshold
        vehicle_front = None
        lane_id = ego_vehicle_lane_center.lane_id - direction
        if lane_id != -1 and lane_id != -2 and lane_id != -3:
            return vehicle_front

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            # check whether in the same road
            target_lane_center = get_lane_center(self._map, target_vehicle.get_location())
            if target_lane_center.transform.location.distance(target_vehicle.get_location()) > target_lane_center.lane_width / 2 + 0.1:
                continue
            if not test_waypoint(target_vehicle_waypoint):
                continue
            # check whether in the specific lane
            if target_vehicle_waypoint.lane_id != lane_id:
                continue
            # if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
            #         target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            #     continue

            loc = target_vehicle.get_location()
            if is_within_distance_ahead(loc, ego_vehicle_location, ego_vehicle_transform, self._proximity_threshold):
                if ego_vehicle_location.distance(loc) < min_distance:
                    # Return the most close vehicel in front of ego vehicle
                    vehicle_front = target_vehicle
                    min_distance = ego_vehicle_location.distance(loc)

        return vehicle_front

    def _is_light_red_us_style(self, lights_list):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
            - bool_flag is True if there is a traffic light in RED
            affecting us and False otherwise
            - traffic_light is the object itself or None if there is no
            red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_intersection:
            # It is too late. Do not block the intersection! Keep going!
            return False

        if self._target_waypoint is not None:
            if self._target_waypoint.is_intersection:
                potential_lights = []
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 80.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.libcarla.TrafficLightState.Red:
                        return True
                else:
                    self._last_traffic_light = None

        return False

    def _retrieve_options(self, list_waypoints, current_waypoint):
        """
        Compute the type of connection between the current active waypoint and the multiple waypoints present in
        list_waypoints. The result is encoded as a list of RoadOption enums.

        :param list_waypoints: list with the possible target waypoints in case of multiple options
        :param current_waypoint: current active waypoint
        :return: list of RoadOption enums representing the type of connection from the active waypoint to each
            candidate in list_waypoints
        """
        options = []
        for next_waypoint in list_waypoints:
            # this is needed because something we are linking to
            # the beggining of an intersection, therefore the
            # variation in angle is small
            next_next_waypoint = next_waypoint.next(3.0)[0]
            link = self._compute_connection(current_waypoint, next_next_waypoint)
            options.append(link)

        return options

    def _compute_connection(self, current_waypoint, next_waypoint):
        """
        Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
        (next_waypoint).

        :param current_waypoint: active waypoint
        :param next_waypoint: target waypoint
        :return: the type of topological connection encoded as a RoadOption enum:
            RoadOption.STRAIGHT
            RoadOption.LEFT
            RoadOption.RIGHT
        """
        n = next_waypoint.transform.rotation.yaw
        n = n % 360.0

        c = current_waypoint.transform.rotation.yaw
        c = c % 360.0

        diff_angle = (n - c) % 180.0
        if diff_angle < 1.0:
            return RoadOption.STRAIGHT
        elif diff_angle > 90.0:
            return RoadOption.LEFT
        else:
            return RoadOption.RIGHT
