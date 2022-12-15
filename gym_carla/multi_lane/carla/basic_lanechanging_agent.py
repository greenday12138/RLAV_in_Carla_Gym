# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
from enum import Enum
from collections import deque
import random
import numpy as np
from shapely.geometry import Polygon

from gym_carla.multi_lane.util.misc import get_speed, draw_waypoints, is_within_distance, get_trafficlight_trigger_location, compute_distance, get_lane_center
from gym_carla.multi_lane.carla.controller import VehiclePIDController

FOLLOW = 0
CHANGE_LEFT = -1
CHANGE_RIGHT = 1

class Basic_Lanechanging_Agent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, target_speed=20, opt_dict={}):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
        """
        self._vehicle = vehicle
        self._vehicle_location = self._vehicle.get_location()
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicle = False
        self._ignore_change_gap = False
        self.lanechanging_fps = 50
        self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        # get the last action of the autonomous vehicle,
        # check whether the autonomous vehicle is during a lane-changing behavior
        self.last_ego_state = FOLLOW
        self.last_lane_id = get_lane_center(self._map, self._vehicle_location).lane_id
        self.lane_change = random.choice([CHANGE_LEFT, FOLLOW, CHANGE_RIGHT])
        self.autopilot_step = 0

        # set by carla_env.py
        self.left_wps = None
        self.center_wps = None
        self.right_wps = None
        self.left_rear_wps = None
        self.center_rear_wps = None
        self.right_rear_wps = None
        self.vehicle_inlane = None

        self.distance_to_left_front = None
        self.distance_to_center_front = None
        self.distance_to_right_front = None
        self.distance_to_left_rear = None
        self.distance_to_center_rear = None
        self.distance_to_right_rear = None

        self.left_next_wayppoint = None
        self.center_next_waypoint = None
        self.right_next_waypoint = None

        self.enable_left_change = True
        self.enable_right_change = True

        # Change parameters according to the dictionary
        opt_dict['target_speed'] = target_speed
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'max_steering' in opt_dict:
            self._max_steering = opt_dict['max_steering']
        if 'max_throttle' in opt_dict:
            self._max_throttle = opt_dict['max_throttle']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'buffer_size' in opt_dict:
            self._buffer_size = opt_dict['buffer_size']
        if 'ignore_front_vehicle' in opt_dict:
            self._ignore_vehicle = opt_dict['ignore_front_vehicle']
        if 'ignore_change_gap' in opt_dict:
            self._ignore_change_gap = opt_dict['ignore_change_gap']
        if 'lanechanging_fps' in opt_dict:
            self.lanechanging_fps = opt_dict['lanechanging_fps']

        print('ignore_front_vehicle, ignore_change_gap: ', self._ignore_vehicle, self._ignore_change_gap)

        self.left_random_change = []
        self.center_random_change = []
        self.right_random_change = []
        self.init_random_change()

        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict)

    def init_random_change(self):
        for i in range(self.lanechanging_fps):
            self.left_random_change.append(0)
            self.center_random_change.append(0)
            # center_random_change.append(0)
            self.right_random_change.append(0)
        self.left_random_change.append(1)
        self.center_random_change.append(1)
        self.center_random_change.append(-1)
        self.right_random_change.append(-1)

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def calculate_distance_to_front(self, wps, vehicle):
        if len(wps) == 0:
            dis = 0
        else:
            dis = self._buffer_size
        if vehicle is not None:
            dis = self.compute_s_distance(wps, vehicle.get_location())
        return dis

    def set_info(self, info_dict):
        """
        :param left_wps: waypoints in left-front lane
        :param center_wps: waypoints in center-front lane
        :param right_wps: waypoints in right-front lane
        :param vehicle_inlane: six vehicles in left-front, center-front, right-front, left-rear, center-rear, right-rear
        :return:
        """
        self.left_wps = info_dict['left_wps']
        self.center_wps = info_dict['center_wps']
        self.right_wps = info_dict['right_wps']
        self.left_rear_wps = info_dict['left_rear_wps']
        self.center_rear_wps = info_dict['center_rear_wps']
        self.right_rear_wps = info_dict['right_rear_wps']
        self.vehicle_inlane = info_dict['vehicle_inlane']
        self._vehicle_location = self._vehicle.get_location()
        # for wps in self.left_wps:
        #     print(wps.transform.location)
        # print('ego_vehicle:', self._vehicle_location)
        # for i in range(6):
        #     v = self.vehicle_inlane[i]
        #     print(v)
        #     if v is not None:
        #         print(i, v.get_location())
        print('the length of six waypoint queues: ', len(self.left_wps), len(self.center_wps), len(self.right_wps), len(self.left_rear_wps),
              len(self.center_rear_wps), len(self.right_rear_wps))
        # For simplicity, we compute s for front vehicles, and compute Euler distance for rear vehicles.
        self.distance_to_left_front = self.calculate_distance_to_front(self.left_wps, self.vehicle_inlane[0])
        self.distance_to_center_front = self.calculate_distance_to_front(self.center_wps, self.vehicle_inlane[1])
        self.distance_to_right_front = self.calculate_distance_to_front(self.right_wps, self.vehicle_inlane[2])
        self.distance_to_left_rear = self.calculate_distance_to_front(self.left_rear_wps, self.vehicle_inlane[3])
        self.distance_to_center_rear = self.calculate_distance_to_front(self.center_rear_wps, self.vehicle_inlane[4])
        self.distance_to_right_rear = self.calculate_distance_to_front(self.right_rear_wps, self.vehicle_inlane[5])
        # print("distance with six vehicles", 'distance_to_left_front: ', self.distance_to_left_front,
        #       'distance_to_center_front: ', self.distance_to_center_front,
        #       'distance_to_right_front: ', self.distance_to_right_front,
        #       'distance_to_left_rear: ', self.distance_to_left_rear,
        #       'distance_to_center_rear: ', self.distance_to_center_rear,
        #       'distance_to_right_rear: ', self.distance_to_right_rear)
        # self.distance_to_left_rear = self._vehicle.get_location().distance(self.vehicle_inlane[3].get_location())
        # self.distance_to_center_rear = self._vehicle.get_location().distance(self.vehicle_inlane[4].get_location())
        # self.distance_to_right_rear = self._vehicle.get_location().distance(self.vehicle_inlane[5].get_location())
        # set next waypoint that distance == 2m
        # if len(self.left_wps) != 0:
        #     self.left_next_wayppoint = self.left_wps[1]
        # if len(self.center_wps) != 0:
        #     self.center_next_waypoint = self.center_wps[1]
        # if len(self.right_wps) != 0:
        #     self.right_next_waypoint = self.right_wps[1]
        self.enable_left_change = self.check_enable_change(self.left_wps, self.distance_to_left_front, self.distance_to_left_rear)
        self.enable_right_change = self.check_enable_change(self.right_wps, self.distance_to_right_front, self.distance_to_right_rear)
        if self._ignore_change_gap:
            self.enable_left_change = True
            self.enable_right_change = True
        else:
            self.enable_left_change = False
            self.enable_right_change = False
            if self.distance_to_left_front / self.distance_to_center_front > 1.1 and self.distance_to_left_rear > 20:
                self.enable_left_change = True
            if self.distance_to_right_front / self.distance_to_center_front > 1.1 and self.distance_to_right_rear > 20:
                self.enable_right_change = True
        print("distance enable: ", self.distance_to_left_front, self.distance_to_center_front,
              self.distance_to_right_front, self.distance_to_left_rear, self.distance_to_center_rear,
              self.distance_to_right_rear, self.enable_left_change, self.enable_right_change)

    def check_enable_change(self, wps_queue, front_distacne, rear_distance):
        """
        check whether enbable a lane-changing behavior
        :param front_distacne:
        :param rear_distance:
        :return:
        """
        enable = False
        V = 10
        T = 1.5
        if len(wps_queue) != 0 and front_distacne >= V * T and rear_distance >= V * T:
            enable = True
        return enable

    def compute_s_distance(self, wps_list, target_location):
        """

        :param wps: 50 waypoints in front
        :param target_location: the position of a surrounding vehicle of the autonomous vehicle
        :return: s
        """
        min_dis = 1000
        s = len(wps_list)
        for i in range(len(wps_list)):
            wps = wps_list[i]
            current_dis = wps.transform.location.distance(target_location)
            if current_dis < min_dis:
                min_dis = current_dis
                s = i + 1
        return s

    def run_step(self, current_lane, target_lane, last_action, under_rl, action, modify_change_steer):
        self.autopilot_step = self.autopilot_step + 1
        """Execute one step of navigation."""
        hazard_detected = False
        # Retrieve all relevant actors
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle = self._vehicle_obstacle_detected(max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        if current_lane == -2:
            self.lane_change = random.choice(self.center_random_change)
        elif current_lane == -1:
            self.lane_change = random.choice(self.left_random_change)
        elif current_lane == -3:
            self.lane_change = random.choice(self.right_random_change)
        else:
            # just to avoid error, dont work
            self.lane_change = 0

        if not under_rl:
            if current_lane == target_lane:
                new_action = self.lane_change
                if new_action == -1 and not self.enable_left_change:
                    new_action = 0
                if new_action == 1 and not self.enable_right_change:
                    new_action = 0
                new_target_lane = current_lane - new_action
            else:
                new_action = last_action
                new_target_lane = target_lane
        else:
            new_action = action - 1
            new_target_lane = target_lane - new_action
        if under_rl:
            control = None
        else:
            control = self._local_planner.run_step({'distance_to_left_front': self.distance_to_left_front,
                                                    'distance_to_center_front': self.distance_to_center_front,
                                                    'distance_to_right_front': self.distance_to_right_front,
                                                    'distance_to_left_rear': self.distance_to_left_rear,
                                                    'distance_to_right_rear': self.distance_to_right_rear,
                                                    'left_wps': self.left_wps,
                                                    'center_wps': self.center_wps,
                                                    'right_wps': self.right_wps,
                                                    'new_action': new_action})
            if modify_change_steer:
                if new_action == -1:
                    control.steer = np.clip(control.steer, -1, 0)
                elif new_action == 1:
                    control.steer = np.clip(control.steer, 0, 1)
            if hazard_detected:
                control = self.add_emergency_stop(control)
        return control, new_target_lane, new_action, [self.distance_to_left_front, self.distance_to_center_front, self.distance_to_right_front], \
               [self.distance_to_left_rear, self.distance_to_center_rear, self.distance_to_right_rear]

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def get_step(self):
        return self.autopilot_step

    def get_follow_action(self):
        pass

    def _vehicle_obstacle_detected(self, max_dis):
        have_dangerous_vehicle = False
        if self.distance_to_center_front < max_dis and not self._ignore_vehicle:
            have_dangerous_vehicle = True

        return have_dangerous_vehicle

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a
    trajectory of waypoints that is generated on-the-fly.

    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice,
    unless a given global plan has already been specified.
    """

    def __init__(self, vehicle, opt_dict={}):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with different parameters:
            dt: time between simulation steps
            target_speed: desired cruise speed in Km/h
            sampling_radius: distance between the waypoints part of the plan
            lateral_control_dict: values of the lateral PID controller
            longitudinal_control_dict: values of the longitudinal PID controller
            max_throttle: maximum throttle applied to the vehicle
            max_brake: maximum brake applied to the vehicle
            max_steering: maximum steering applied to the vehicle
            offset: distance between the route waypoints and the center of the lane
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._vehicle_controller = None
        self.target_waypoint = None
        self.target_road_option = None

        self._waypoints_queue = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False

        # Base parameters
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = 2.0
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self._dt}
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0
        self._base_min_distance = 3.0
        self._follow_speed_limits = False

        # Overload parameters
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = opt_dict['sampling_radius']
            if 'lateral_control_dict' in opt_dict:
                self._args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                self._args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'max_throttle' in opt_dict:
                self._max_throt = opt_dict['max_throttle']
            if 'max_brake' in opt_dict:
                self._max_brake = opt_dict['max_brake']
            if 'max_steering' in opt_dict:
                self._max_steer = opt_dict['max_steering']
            if 'offset' in opt_dict:
                self._offset = opt_dict['offset']
            if 'base_min_distance' in opt_dict:
                self._base_min_distance = opt_dict['base_min_distance']
            if 'follow_speed_limits' in opt_dict:
                self._follow_speed_limits = opt_dict['follow_speed_limits']

        # initializing controller
        self._init_controller()

    def reset_vehicle(self):
        """Reset the ego-vehicle"""
        self._vehicle = None

    def _init_controller(self):
        """Controller initialization"""
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

        # Compute the current vehicle waypoint
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def run_step(self, opt_dict):
        # distance_to_left_front = opt_dict['distance_to_left_front']
        # distance_to_center_front = opt_dict['distance_to_center_front']
        # distance_to_right_front = opt_dict['distance_to_right_front']
        # distance_to_left_rear = opt_dict['distance_to_left_rear']
        # distance_to_right_rear = opt_dict['distance_to_right_rear']
        left_wps = opt_dict['left_wps']
        center_wps = opt_dict['center_wps']
        right_wps = opt_dict['right_wps']
        new_action = opt_dict['new_action']
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        veh_waypoint = get_lane_center(self._map, veh_location)

        vehicle_speed = get_speed(self._vehicle) / 3.6
        lane_center_ratio = 1 - veh_waypoint.transform.location.distance(veh_location) / 4
        self._min_distance = self._base_min_distance * lane_center_ratio
        print('min_distance: ', self._min_distance)
        next_wp = 1
        if self._min_distance > 1:
            next_wp = 2
        elif self._min_distance > 2:
            next_wp = 3
        elif self._min_distance > 3:
            next_wp = 4
        if new_action == -1:
            self.target_waypoint = left_wps[next_wp+10-1]
            # print('left target waypoint: ', self.target_waypoint)
        elif new_action == 0:
            self.target_waypoint = center_wps[next_wp+2-1]
            # print('center target waypoint: ', self.target_waypoint)
        elif new_action == 1:
            self.target_waypoint = right_wps[next_wp+10-1]
            # print('right target waypoint: ', self.target_waypoint)
        # print("current location and target location: ", veh_location, self.target_waypoint.transform.location)
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        return control


