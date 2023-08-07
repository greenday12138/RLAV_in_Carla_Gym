import carla
import math, random
import numpy as np
from macad_gym.core.controllers.local_planner import LocalPlanner
from macad_gym.core.utils.misc import (get_speed, get_yaw_diff, draw_waypoints, get_lane_center,
                                       get_projection, compute_signed_distance)

class StateDAO(object):
    # class for gettting surrounding information
    def __init__(self, configs):
        self._scenario_config = configs["scenario_config"]
        self._env_config = configs["env_config"]
        self._actor_configs = configs["actor_config"]
        self._rl_configs = configs["rl_config"]
        self._local_planner = {}
        self._actors = configs["actors"]
        self.world = configs["world"]
        self.map = configs["map"]
        self._cur_measurement = {}
        for actor_id, actor_config in self._actor_configs.items():
            self._local_planner[actor_id] = LocalPlanner(
                self._actors[actor_id], {
                    'sampling_resolution': self._env_config["sampling_resolution"],
                    'buffer_size': self._env_config["buffer_size"],
                    'vehicle_proximity': self._env_config["vehicle_proximity"],
                    'traffic_light_proximity':self._env_config["traffic_light_proximity"]})

    def get_state(self, actor_id):
        wps_info, lights_info, vehs_info = self._local_planner[actor_id].run_step()
        if self._rl_configs["debug"]:
            draw_waypoints(self.world, wps_info.center_front_wps+wps_info.center_rear_wps+\
                wps_info.left_front_wps+wps_info.left_rear_wps+wps_info.right_front_wps+wps_info.right_rear_wps, 
                self._env_config["fixed_delta_seconds"] + 0.001, z=1,
                color=random.choice([(255,0,0)]))

        left_wps=wps_info.left_front_wps
        center_wps=wps_info.center_front_wps
        right_wps=wps_info.right_front_wps

        lane_center = get_lane_center(self.map, self._actors[actor_id].get_location())
        ego_t = compute_signed_distance(lane_center.transform.location,
                                        self._actors[actor_id].get_location(),
                                        lane_center.transform.get_forward_vector())
        # right_lane_dis = lane_center.get_right_lane(
        #     ).transform.location.distance(self._actors[actor_id].get_location())
        # ego_t= lane_center.lane_width / 2 + lane_center.get_right_lane().lane_width / 2 - right_lane_dis

        hero_vehicle_z = lane_center.transform.location.z
        ego_forward_vector = self._actors[actor_id].get_transform().get_forward_vector()
        my_sample_ratio = self._env_config["buffer_size"] // 10
        center_wps_processed = process_lane_wp(center_wps, hero_vehicle_z, ego_forward_vector, my_sample_ratio, 0)
        if len(left_wps) == 0:
            left_wps_processed = center_wps_processed.copy()
            for left_wp in left_wps_processed:
                left_wp[2] = -1
        else:
            left_wps_processed = process_lane_wp(left_wps, hero_vehicle_z, ego_forward_vector, my_sample_ratio, -1)
        if len(right_wps) == 0:
            right_wps_processed = center_wps_processed.copy()
            for right_wp in right_wps_processed:
                right_wp[2] = 1
        else:
            right_wps_processed = process_lane_wp(right_wps, hero_vehicle_z, ego_forward_vector, my_sample_ratio, 1)

        left_wall = False
        if len(left_wps) == 0:
            left_wall = True
        right_wall = False
        if len(right_wps) == 0:
            right_wall = True
        vehicle_inlane_processed = self._process_veh(self._actors[actor_id
                    ],vehs_info, left_wall, right_wall, self._env_config["vehicle_proximity"])

        yaw_diff_ego = math.degrees(get_yaw_diff(lane_center.transform.get_forward_vector(),
                                    self._actors[actor_id].get_transform().get_forward_vector()))

        yaw_forward = lane_center.transform.get_forward_vector()
        v_3d = self._actors[actor_id].get_velocity()
        v_s,v_t=get_projection(v_3d,yaw_forward)

        a_3d = self._actors[actor_id].get_acceleration()
        a_s,a_t=get_projection(a_3d,yaw_forward)

        if lights_info:
            wps=lights_info.get_stop_waypoints()
            stop_dis=1.0
            for wp in wps:
                if wp.road_id==lane_center.road_id and wp.lane_id==lane_center.lane_id:
                    stop_dis=wp.transform.location.distance(lane_center.transform.location
                                                            )/self._env_config["traffic_light_proximity"]
                    break
            if (lights_info.state==carla.TrafficLightState.Red or lights_info.state==carla.TrafficLightState.Yellow):
                light=[0,1,stop_dis]
            else:
                light=[1,0,stop_dis]
        else:
            stop_dis=1.0
            light=[1,0,stop_dis]

        """Attention:
        Upon initializing, there are some bugs in the theta_v and theta_a, which could be greater than 90,
        this might be caused by carla."""
        self._cur_measurement["velocity"] = v_s
        self._cur_measurement["current_acc"] = a_s
        #update informatino for rear vehicle
        if vehs_info.center_rear_veh is None or \
                (lights_info is not None and lights_info.state!=carla.TrafficLightState.Green):
            self._cur_measurement["rear_id"] = -1
            self._cur_measurement["rear_v"] = 0
            self._cur_measurement["rear_a"] = 0
            self._cur_measurement["change_lane"] = None
        else:
            lane_center=get_lane_center(self.map, vehs_info.center_rear_veh.get_location())
            yaw_forward=lane_center.transform.get_forward_vector()
            v_3d=vehs_info.center_rear_veh.get_velocity()
            v_s,v_t=get_projection(v_3d,yaw_forward)
            a_3d=vehs_info.center_rear_veh.get_acceleration()
            a_s,a_t=get_projection(a_3d,yaw_forward)
            self._cur_measurement["rear_id"] = vehs_info.center_rear_veh.id
            self._cur_measurement["rear_v"] = v_s
            self._cur_measurement["rear_a"] = a_s
            self._cur_measurement["change_lane"] = None

        return ({
                "wps": wps_info,
                "lights": lights_info,
                "vehs": vehs_info
            },
            self._cur_measurement,
            {
                "left_waypoints": left_wps_processed, 
                "center_waypoints": center_wps_processed,
                "right_waypoints": right_wps_processed, 
                "vehicle_info": vehicle_inlane_processed,
                "hero_vehicle": [v_s/10, v_t/10, a_s/3, a_t/3, ego_t, yaw_diff_ego/90],
                "light":light
            }
        ) 
    
    def _process_veh(self, ego_vehicle, vehs_info, left_wall, right_wall,vehicle_proximity):        
        vehicle_inlane=[vehs_info.left_front_veh,vehs_info.center_front_veh,vehs_info.right_front_veh,
                vehs_info.left_rear_veh,vehs_info.center_rear_veh,vehs_info.right_rear_veh]
        vehicle_distance_s=[vehs_info.distance_to_front_vehicles[0], vehs_info.distance_to_front_vehicles[1],
                          vehs_info.distance_to_front_vehicles[2], vehs_info.distance_to_rear_vehicles[0],
                          vehs_info.distance_to_rear_vehicles[1], vehs_info.distance_to_rear_vehicles[2]]

        all_v_info = []
        #print('vehicle_inlane: ', vehicle_inlane)
        for i in range(6):
            if i == 0 or i == 3:
                lane = -1
            elif i == 1 or i == 4:
                lane = 0
            else:
                lane = 1
            veh = vehicle_inlane[i]
            wall = False
            if left_wall and (i == 0 or i == 3):
                wall = True
            if right_wall and (i == 2 or i == 5):
                wall = True
            if wall:
                if i < 3:
                    #v_info = [0.001, 0, lane]
                    v_info = [0.001, 0.001, 0, lane]
                else:
                    #v_info = [-0.001, 0, lane]
                    v_info = [-0.001, 0.001, 0, lane]
            else:
                if veh is None:
                    if i < 3:
                        #v_info = [1, 0, lane]
                        v_info = [1, 1, 0, lane]
                    else:
                        #v_info = [-1, 0, lane]
                        v_info = [-1, 1, 0, lane]
                else:
                    ego_speed = get_speed(ego_vehicle, False)
                    ego_half_len, ego_half_wid = get_len_wid(ego_vehicle)
                    veh_speed = get_speed(veh, False)
                    rel_speed = ego_speed - veh_speed

                    # ego_bounding_x = ego_vehicle.bounding_box.extent.x
                    # ego_bounding_y = ego_vehicle.bounding_box.extent.y
                    # distance = ego_location.distance(veh.get_location())
                    # vehicle_len = max(abs(ego_bounding_x), abs(ego_bounding_y)) + \
                    #     max(abs(veh.bounding_box.extent.x), abs(veh.bounding_box.extent.y))
                    # distance -= vehicle_len
                    distance = vehicle_distance_s[i]
                    veh_half_len, veh_half_wid = get_len_wid(veh)
                    distance -= veh_half_len + ego_half_len
                    
                    # compute lateral distance -- distance_t
                    veh_lane_center = get_lane_center(self.map, vehicle_inlane[i].get_location())
                    veh_lcen = compute_signed_distance(veh_lane_center.transform.location,
                                                       vehicle_inlane[i].get_location(),
                                                       veh_lane_center.transform.get_forward_vector())
                    ego_lane_center = get_lane_center(self.map, ego_vehicle.get_location())
                    lane_wid = ego_lane_center.lane_width
                    ego_lcen = compute_signed_distance(ego_lane_center.transform.location, 
                                                       ego_vehicle.get_location(),
                                                       ego_lane_center.transform.get_forward_vector())
                    if i == 0 or i == 3:
                        distance_t = (lane_wid - (veh_lcen + veh_half_wid - ego_lcen + ego_half_wid)) / lane_wid
                    elif i == 2 or i == 5:
                        distance_t = (lane_wid - ( -veh_lcen + veh_half_wid + ego_lcen + ego_half_wid)) / lane_wid
                    else:
                        # i == 1 or i == 4
                        distance_t = (abs(ego_lcen - veh_lcen) - lane_wid) / lane_wid

                    if distance < 0:
                        if i < 3:
                            #v_info = [0.001, rel_speed, lane]
                            v_info = [0.001, distance_t, rel_speed, lane]
                        else:
                            #v_info = [-0.001, -rel_speed, lane]
                            v_info = [-0.001, distance_t, -rel_speed, lane]
                    else:
                        if i < 3:
                            v_info = [distance / vehicle_proximity, distance_t, rel_speed, lane]
                        else:
                            v_info = [-distance / vehicle_proximity, distance_t, -rel_speed, lane]

            # # remove lateral distance
            # v_info[1] = 0
            all_v_info.append(v_info)
        # print(all_v_info)
        return np.array(all_v_info)


def process_lane_wp(wps_list, ego_vehicle_z, ego_forward_vector, my_sample_ratio, lane_offset):
    wps = []
    idx = 0

    # for wp in wps_list:
    #     delta_z = wp.transform.location.z - ego_vehicle_z
    #     yaw_diff = math.degrees(get_yaw_diff(wp.transform.get_forward_vector(), ego_forward_vector))
    #     yaw_diff = yaw_diff / 90
    #     if idx % my_sample_ratio == my_sample_ratio-1:
    #         wps.append([delta_z/2, yaw_diff, lane_offset])
    #     idx = idx + 1
    # return np.array(wps)
    for i in range(10):
        wp = wps_list[i]
        delta_z = wp.transform.location.z - ego_vehicle_z
        yaw_diff = math.degrees(get_yaw_diff(wp.transform.get_forward_vector(), ego_forward_vector))
        yaw_diff = yaw_diff / 90
        wps.append([delta_z/3, yaw_diff, lane_offset])
    return np.array(wps)

def get_len_wid(vehicle):
    proj_s, proj_t = get_projection(vehicle.bounding_box.extent, 
                                vehicle.bounding_box.rotation.get_forward_vector())
    
    return abs(proj_s), abs(proj_t)