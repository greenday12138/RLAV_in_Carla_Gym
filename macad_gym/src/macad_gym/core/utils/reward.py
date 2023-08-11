import math
import carla
import numpy as np
from macad_gym.viz.logger import LOG
from macad_gym.core.utils.misc import (get_speed, get_yaw_diff, get_sign, test_waypoint,
                                       get_lane_center, get_projection, compute_signed_distance)
from macad_gym.core.utils.wrapper import SemanticTags, Truncated, Action


def get_len_wid(vehicle):
    proj_s, proj_t = get_projection(vehicle.bounding_box.extent, 
                                vehicle.bounding_box.rotation.get_forward_vector())
    
    return abs(proj_s), abs(proj_t)

class Reward(object):
    def __init__(self, configs):
        self.reward = 0.0
        self.prev = {}
        self.curr = {}

    def set_state(self, actor, state, map):
        pass

    def compute_reward(self, actor_id, prev_measurement, curr_measurement, flag):
        self.prev = prev_measurement
        self.curr = curr_measurement

        if flag == "corl2017":
            return self.compute_reward_corl2017()
        elif flag == "lane_keep":
            return self.compute_reward_lane_keep()
        elif flag == "custom":
            return self.compute_reward_custom()

    def compute_reward_custom(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0

        self.reward -= self.curr["intersection_offroad"] * 0.05
        self.reward -= self.curr["intersection_otherlane"] * 0.05

        if self.curr["next_command"] == "REACH_GOAL":
            self.reward += 100

        return self.reward

    def compute_reward_corl2017(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        # Change in speed (km/h)
        self.reward += 0.05 * (
            self.curr["velocity"] - self.prev["velocity"])
        # New collision damage
        self.reward -= .00002 * (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])

        # New sidewalk intersection
        self.reward -= 2 * (self.curr["intersection_offroad"] -
                            self.prev["intersection_offroad"])

        # New opposite lane intersection
        self.reward -= 2 * (self.curr["intersection_otherlane"] -
                            self.prev["intersection_otherlane"])

        return self.reward

    def compute_reward_lane_keep(self):
        self.reward = 0.0
        # Speed reward, up 30.0 (km/h)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        # New collision damage
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0
        # Sidewalk intersection
        self.reward -= self.curr["intersection_offroad"]
        # Opposite lane intersection
        self.reward -= self.curr["intersection_otherlane"]

        return self.reward

    def info(self):
        return {}

    def destory(self):
        pass



class SACReward(Reward):
    def __init__(self, configs):
        self._scenario_config = configs["scenarios"]
        self._env_config = configs["env"]
        self._actor_configs = configs["actors"]
        self._rl_configs = configs["rl_parameters"]
        self.TTC_THRESHOLD = 6.0001
        self.reward = 0.0
        self.ttc_reward = 0.0
        self.efficiency_reward = 0.0
        self.comfort_reward = 0.0
        self.lane_center_reward = 0.0
        self.lane_change_reward = 0.0
        self.prev = {}
        self.curr = {}
        self.state = {}
        self.vehicle = None

    def set_state(self, actor, state, map):
        self.vehicle = actor
        self.state = state
        self.map = map

    def compute_reward(self, actor_id, prev_measurement, curr_measurement, flag):
        self.reward = None
        self.actor_id = actor_id
        self.prev = prev_measurement
        self.curr = curr_measurement

        truncated = self.curr["truncated"]
        abandon = False
        if truncated != str(Truncated.FALSE):
            if truncated == str(Truncated.COLLISION):
                if self.curr["collision_vehicles"] > 0 or self.curr["collision_pedestrians"] > 0:
                    self.reward = -self._rl_configs["penalty"]
                else:
                    # Abandon the experience that ego vehicle collide with other obstacle
                    abandon = True
            else:
                self.reward = -self._rl_configs["penalty"]

        lane_center = get_lane_center(self.map, self.vehicle.get_location())
        yaw_forward = lane_center.transform.get_forward_vector().make_unit_vector()

        ttc, self.ttc_reward = self._ttc_reward(self.state["vehs"].center_front_veh)
        self.efficiency_reward = self._efficiency_reward(yaw_forward) * 2
        self.comfort_reward, yaw_change = self._comfort_reward(yaw_forward)
        Lcen, self.lane_center_reward = self._lane_center_reward(lane_center)
        self.lane_change_reward = self._lane_change_reward()
        if self.reward is None:
            self.reward = self.ttc_reward + self.lane_center_reward + self.lane_change_reward + \
                self.efficiency_reward
        
        self.vehicle = None
        self.state = {}
        self.prev = {}
        self.curr = {}

        return self.reward

    def info(self):
        return {
            "ttc_reward": self.ttc_reward,
            "efficiency_reward": self.efficiency_reward,
            "lane_center_reward": self.lane_center_reward,
            "comfort_reward": self.comfort_reward,
            "lane_change_reward": self.lane_change_reward,
        }

    def _comfort_reward(self, yaw_forward):
        a_3d = self.vehicle.get_acceleration()
        cur_acc, a_t = get_projection(a_3d, yaw_forward)
        fps = 1 / self._env_config["fixed_delta_seconds"]
        acc_jerk = - \
            ((self.curr["current_acc"] - self.curr["last_acc"])
             * (fps)) ** 2 / ((6 * fps) ** 2)
        last_yaw = carla.Vector3D(x=self.curr["last_yaw"]["x"], y=self.curr["last_yaw"]["y"],
                                  z=self.curr["last_yaw"]["z"])
        yaw_diff = math.degrees(get_yaw_diff(last_yaw, yaw_forward))
        Yaw_jerk = -abs(yaw_diff) / 30

        return np.clip(acc_jerk * 0.5 + Yaw_jerk, -0.5, 0), yaw_diff

    def _efficiency_reward(self, yaw_forward):
        v_3d = self.vehicle.get_velocity()
        v_s, v_t = get_projection(v_3d, yaw_forward)
        speed_1, speed_2 = self._actor_configs[self.actor_id]["speed_limit"], \
            self._actor_configs[self.actor_id]["speed_limit"]
        # if self.lights_info and self.lights_info.state!=carla.TrafficLightState.Green:
        #     wps=self.lights_info.get_stop_waypoints()
        #     for wp in wps:
        #         if wp.lane_id==lane_center.lane_id:
        #             dis=self.ego_vehicle.get_location().distance(wp.transform.location)
        #             if dis<self.traffic_light_proximity:
        #                 speed_1=(dis+0.0001)/self.traffic_light_proximity*self.speed_limit
        max_speed = min(speed_1, speed_2)
        if v_s * 3.6 > max_speed:
            # fEff = 1
            fEff = math.exp(max_speed - v_s * 3.6) - 1
        else:
            fEff = v_s * 3.6 / max_speed - 1
        # if max_speed<self.speed_min:
        #     fEff=1

        return fEff

    def _ttc_reward(self, target_veh):
        """Caculate the time left before ego vehicle collide with target vehicle"""
        # TTC = float('inf')
        TTC = self.TTC_THRESHOLD
        ego_veh = self.vehicle
        if target_veh and ego_veh:
            distance = ego_veh.get_location().distance(target_veh.get_location())
            ego_half_len, ego_half_wid = get_len_wid(ego_veh)
            veh_half_len, veh_half_wid = get_len_wid(target_veh)
            # vehicle_len = max(abs(ego_veh.bounding_box.extent.x),
            #                   abs(ego_veh.bounding_box.extent.y)) + \
            #     max(abs(target_veh.bounding_box.extent.x),
            #         abs(target_veh.bounding_box.extent.y))
            distance -= ego_half_len + veh_half_len
            # rel_speed = get_speed(ego_veh,False) - get_speed(target_veh, False)
            # if abs(rel_speed) > float(0.0000001):
            #     TTC = distance / rel_speed
            if distance < self._env_config["min_distance"]:
                TTC = 0.01
            else:
                distance -= self._env_config["min_distance"]
                rel_speed = get_speed(ego_veh, False) - \
                    get_speed(target_veh, False)
                if abs(rel_speed) > float(0.0000001):
                    TTC = distance / rel_speed
        # fTTC=-math.exp(-TTC)
        if TTC >= 0 and TTC <= self.TTC_THRESHOLD:
            fTTC = TTC / self.TTC_THRESHOLD - 1
            #fTTC = np.clip(np.log(TTC / self.TTC_THRESHOLD), -1, 0)
        else:
            fTTC = 0
            # TTC=TTC_THRESHOLD

        return TTC, fTTC

    def _lane_center_reward(self, lane_center):
        if not test_waypoint(lane_center, True):
            Lcen = 2.1
            fLcen = -2
            LOG.reward_logger.debug(f"lane_center.lane_id:{lane_center.lane_id}, lane_center.road_id:{lane_center.road_id}, "
                         f"flcen:{fLcen}, lane_wid/2:{lane_center.lane_width / 2}")
        else:
            Lcen = compute_signed_distance(lane_center.transform.location, 
                                           self.vehicle.get_location(),
                                           lane_center.transform.get_forward_vector())
            fLcen = -abs(Lcen)/(lane_center.lane_width/2)
            # if self.current_action == Action.LANE_CHANGE_LEFT and self.current_lane == self.last_lane:
            #     # change left
            #     center_width=lane_center.lane_width
            #     lane_center=lane_center.get_left_lane()
            #     if lane_center is None:
            #         Lcen = 7
            #         fLcen = -2
            #     else:
            #         Lcen =compute_signed_distance(lane_center.transform.location,
            #                                       self.vehicle.get_location(),
            #                                       lane_center.transform.get_forward_vector())
            #         fLcen = -abs(Lcen) / (lane_center.lane_width/2+center_width)
            # elif self.current_action == Action.LANE_CHANGE_RIGHT and self.current_lane == self.last_lane:
            #     #change right
            #     center_width=lane_center.lane_width
            #     lane_center=lane_center.get_right_lane()
            #     if lane_center is None:
            #         Lcen = 7
            #         fLcen = -2
            #     else:
            #         Lcen =compute_signed_distance(lane_center.transform.location,
            #                                       self.vehicle.get_location(),
            #                                       lane_center.transform.get_forward_vector())
            #         fLcen=-abs(Lcen)/(lane_center.lane_width/2+center_width)
            # else:
            #     #lane follow and stop mode
            #     Lcen =compute_signed_distance(lane_center.transform.location,
            #                                   ego_location,
            #                                   lane_center.transform.get_forward_vector())
            #     fLcen = -abs(Lcen)/(lane_center.lane_width/2)
            # LOG.reward_logger.debug('pdqn_lane_center: Lcen, fLcen: ', Lcen, fLcen)
        return Lcen, fLcen

    def _lane_change_reward(self):
        distance_to_front_vehicles, distance_to_rear_vehicles = \
            self.state["vehs"].distance_to_front_vehicles, self.state["vehs"].distance_to_rear_vehicles
        last_lane, current_lane = self.curr["last_lane"], self.curr["current_lane"]
        LOG.reward_logger.debug(f"distance_to_front_vehicles:{distance_to_front_vehicles}, distance_to_rear_vehicles:{distance_to_rear_vehicles}")
        # still the distances of the last time step
        if current_lane - last_lane == 0:
            reward = 0
            ttc, rear_ttc_reward = self._ttc_reward(self.state["vehs"].center_rear_veh)
            LOG.reward_logger.debug(f"lane_change_reward:{reward}, rear_ttc_reward:{rear_ttc_reward}")
        elif current_lane - last_lane == -1:
            # change right
            self.calculate_impact = 1
            center_front_dis = distance_to_front_vehicles[0]
            right_front_dis = distance_to_front_vehicles[1]
            dis=right_front_dis-center_front_dis
            reward=dis/self._env_config["vehicle_proximity"]*self._rl_configs["lane_change_reward"]
            # if right_front_dis > center_front_dis:
            #     reward = min((right_front_dis / center_front_dis - 1) * self.lane_change_reward, self.lane_change_reward)
            # else:
            #     reward = max((right_front_dis / center_front_dis - 1) * self.lane_change_reward, -self.lane_change_reward)
                # reward = 0
            ttc,rear_ttc_reward = self._ttc_reward(self.state["vehs"].center_rear_veh)
            # add rear_ttc_reward?
            LOG.reward_logger.debug(f"lane_change_reward:{reward}, rear_ttc_reward:{rear_ttc_reward}")
        elif current_lane - last_lane == 1:
            # change left
            self.calculate_impact = -1
            center_front_dis = distance_to_front_vehicles[2]
            left_front_dis = distance_to_front_vehicles[1]
            dis=left_front_dis-center_front_dis
            reward=dis/self._env_config["vehicle_proximity"]*self._rl_configs["lane_change_reward"]
            # if left_front_dis > center_front_dis:
            #     reward = min((left_front_dis / center_front_dis - 1) * self.lane_change_reward, self.lane_change_reward)
            # else:
            #     reward = max((left_front_dis / center_front_dis - 1) * self.lane_change_reward, -self.lane_change_reward)
                # reward = 0
            ttc,rear_ttc_reward = self._ttc_reward(self.state["vehs"].center_rear_veh)
            LOG.reward_logger.debug(f"lane_change_reward:{reward}, rear_ttc_reward:{rear_ttc_reward}")

        return reward



class PDQNReward(Reward):
    def __init__(self, configs):
        self._scenario_config = configs["scenarios"]
        self._env_config = configs["env"]
        self._actor_configs = configs["actors"]
        self._rl_configs = configs["rl_parameters"]
        self.TTC_THRESHOLD = 6.0001
        self.reward = 0.0
        self.ttc_reward = 0.0
        self.efficiency_reward = 0.0
        self.comfort_reward = 0.0
        self.lane_center_reward = 0.0
        self.lane_change_reward = 0.0
        self.prev = {}
        self.curr = {}
        self.state = {}
        self.vehicle = None

    def set_state(self, actor, state, map):
        self.vehicle = actor
        self.state = state
        self.map = map

    def compute_reward(self, actor_id, prev_measurement, curr_measurement, flag):
        self.reward = None
        self.actor_id = actor_id
        self.prev = prev_measurement
        self.curr = curr_measurement

        truncated = self.curr["truncated"]
        abandon = False
        if truncated != str(Truncated.FALSE):
            if truncated == str(Truncated.CHANGE_LANE_IN_LANE_FOLLOW):
                self.reward = -self._rl_configs["lane_penalty"]
            elif truncated == str(Truncated.COLLISION):
                if self.curr["collision_vehicles"] > 0 or self.curr["collision_pedestrians"] > 0:
                    self.reward = -self._rl_configs["penalty"]
                else:
                    # Abandon the experience that ego vehicle collide with other obstacle
                    abandon = True
            else:
                self.reward = -self._rl_configs["penalty"]

        lane_center = get_lane_center(self.map, self.vehicle.get_location())
        yaw_forward = lane_center.transform.get_forward_vector().make_unit_vector()

        ttc, self.ttc_reward = self._ttc_reward(self.state["vehs"].center_front_veh)
        self.efficiency_reward = self._efficiency_reward(yaw_forward) * 2
        self.comfort_reward, yaw_change = self._comfort_reward(yaw_forward)
        Lcen, self.lane_center_reward = self._lane_center_reward(lane_center)
        self.lane_change_reward = self._lane_change_reward()
        if self.reward is None:
            self.reward = self.ttc_reward + self.lane_center_reward + self.lane_change_reward + \
                self.efficiency_reward
        
        self.vehicle = None
        self.state = {}
        self.prev = {}
        self.curr = {}

        return self.reward

    def info(self):
        return {
            "ttc_reward": self.ttc_reward,
            "efficiency_reward": self.efficiency_reward,
            "lane_center_reward": self.lane_center_reward,
            "comfort_reward": self.comfort_reward,
            "lane_change_reward": self.lane_change_reward,
        }

    def _comfort_reward(self, yaw_forward):
        a_3d = self.vehicle.get_acceleration()
        cur_acc, a_t = get_projection(a_3d, yaw_forward)
        fps = 1 / self._env_config["fixed_delta_seconds"]
        acc_jerk = - \
            ((self.curr["current_acc"] - self.curr["last_acc"])
             * (fps)) ** 2 / ((6 * fps) ** 2)
        last_yaw = carla.Vector3D(x=self.curr["last_yaw"]["x"], y=self.curr["last_yaw"]["y"],
                                  z=self.curr["last_yaw"]["z"])
        yaw_diff = math.degrees(get_yaw_diff(last_yaw, yaw_forward))
        Yaw_jerk = -abs(yaw_diff) / 30

        return np.clip(acc_jerk * 0.5 + Yaw_jerk, -0.5, 0), yaw_diff

    def _efficiency_reward(self, yaw_forward):
        v_3d = self.vehicle.get_velocity()
        v_s, v_t = get_projection(v_3d, yaw_forward)
        speed_1, speed_2 = self._actor_configs[self.actor_id]["speed_limit"], \
            self._actor_configs[self.actor_id]["speed_limit"]
        # if self.lights_info and self.lights_info.state!=carla.TrafficLightState.Green:
        #     wps=self.lights_info.get_stop_waypoints()
        #     for wp in wps:
        #         if wp.lane_id==lane_center.lane_id:
        #             dis=self.ego_vehicle.get_location().distance(wp.transform.location)
        #             if dis<self.traffic_light_proximity:
        #                 speed_1=(dis+0.0001)/self.traffic_light_proximity*self.speed_limit
        max_speed = min(speed_1, speed_2)
        if v_s * 3.6 > max_speed:
            # fEff = 1
            fEff = math.exp(max_speed - v_s * 3.6) - 1
        else:
            fEff = v_s * 3.6 / max_speed - 1
        # if max_speed<self.speed_min:
        #     fEff=1

        return fEff

    def _ttc_reward(self, target_veh):
        """Caculate the time left before ego vehicle collide with target vehicle"""
        # TTC = float('inf')
        TTC = self.TTC_THRESHOLD
        ego_veh = self.vehicle
        if target_veh and ego_veh:
            distance = ego_veh.get_location().distance(target_veh.get_location())
            ego_half_len, ego_half_wid = get_len_wid(ego_veh)
            veh_half_len, veh_half_wid = get_len_wid(target_veh)
            # vehicle_len = max(abs(ego_veh.bounding_box.extent.x),
            #                   abs(ego_veh.bounding_box.extent.y)) + \
            #     max(abs(target_veh.bounding_box.extent.x),
            #         abs(target_veh.bounding_box.extent.y))
            distance -= ego_half_len + veh_half_len
            # rel_speed = get_speed(ego_veh,False) - get_speed(target_veh, False)
            # if abs(rel_speed) > float(0.0000001):
            #     TTC = distance / rel_speed
            if distance < self._env_config["min_distance"]:
                TTC = 0.01
            else:
                distance -= self._env_config["min_distance"]
                rel_speed = get_speed(ego_veh, False) - \
                    get_speed(target_veh, False)
                if abs(rel_speed) > float(0.0000001):
                    TTC = distance / rel_speed
        # fTTC=-math.exp(-TTC)
        if TTC >= 0 and TTC <= self.TTC_THRESHOLD:
            fTTC = TTC / self.TTC_THRESHOLD - 1
            #fTTC = np.clip(np.log(TTC / self.TTC_THRESHOLD), -1, 0)
        else:
            fTTC = 0
            # TTC=TTC_THRESHOLD

        return TTC, fTTC

    def _lane_center_reward(self, lane_center):
        if not test_waypoint(lane_center, True):
            Lcen = 2.1
            fLcen = -2
            LOG.reward_logger.debug(f"lane_center.lane_id:{lane_center.lane_id}, lane_center.road_id:{lane_center.road_id}, "
                         f"flcen:{fLcen}, lane_wid/2:{lane_center.lane_width / 2}")
        else:
            Lcen = compute_signed_distance(lane_center.transform.location, 
                                           self.vehicle.get_location(),
                                           lane_center.transform.get_forward_vector())
            fLcen = -abs(Lcen)/(lane_center.lane_width/2)
            # if self.current_action == Action.LANE_CHANGE_LEFT and self.current_lane == self.last_lane:
            #     # change left
            #     center_width=lane_center.lane_width
            #     lane_center=lane_center.get_left_lane()
            #     if lane_center is None:
            #         Lcen = 7
            #         fLcen = -2
            #     else:
            #         Lcen =compute_signed_distance(lane_center.transform.location,
            #                                       self.vehicle.get_location(),
            #                                       lane_center.transform.get_forward_vector())
            #         fLcen = -abs(Lcen) / (lane_center.lane_width/2+center_width)
            # elif self.current_action == Action.LANE_CHANGE_RIGHT and self.current_lane == self.last_lane:
            #     #change right
            #     center_width=lane_center.lane_width
            #     lane_center=lane_center.get_right_lane()
            #     if lane_center is None:
            #         Lcen = 7
            #         fLcen = -2
            #     else:
            #         Lcen =compute_signed_distance(lane_center.transform.location,
            #                                       self.vehicle.get_location(),
            #                                       lane_center.transform.get_forward_vector())
            #         fLcen=-abs(Lcen)/(lane_center.lane_width/2+center_width)
            # else:
            #     #lane follow and stop mode
            #     Lcen =compute_signed_distance(lane_center.transform.location,
            #                                   ego_location,
            #                                   lane_center.transform.get_forward_vector())
            #     fLcen = -abs(Lcen)/(lane_center.lane_width/2)
            # LOG.reward_logger.debug('pdqn_lane_center: Lcen, fLcen: ', Lcen, fLcen)
        return Lcen, fLcen

    def _lane_change_reward(self):
        distance_to_front_vehicles, distance_to_rear_vehicles = \
            self.state["vehs"].distance_to_front_vehicles, self.state["vehs"].distance_to_rear_vehicles
        last_lane, current_lane = self.curr["last_lane"], self.curr["current_lane"]
        LOG.reward_logger.debug(f"distance_to_front_vehicles:{distance_to_front_vehicles}, distance_to_rear_vehicles:{distance_to_rear_vehicles}")
        # still the distances of the last time step
        reward = 0
        if self.curr["current_action"] == str(Action.LANE_FOLLOW):
            # if change lane in lane following mode, we set this reward=0, but will be truncated
            return reward
        if current_lane - last_lane == -1:
            # change right
            self.calculate_impact = 1
            center_front_dis = distance_to_front_vehicles[0]
            right_front_dis = distance_to_front_vehicles[1]
            dis=right_front_dis-center_front_dis
            reward=dis/self._env_config["vehicle_proximity"]*self._rl_configs["lane_change_reward"]
            # if right_front_dis > center_front_dis:
            #     reward = min((right_front_dis / center_front_dis - 1) * self.lane_change_reward, self.lane_change_reward)
            # else:
            #     reward = max((right_front_dis / center_front_dis - 1) * self.lane_change_reward, -self.lane_change_reward)
                # reward = 0
            ttc,rear_ttc_reward = self._ttc_reward(self.state["vehs"].center_rear_veh)
            # add rear_ttc_reward?
            LOG.reward_logger.debug(f"lane_change_reward:{reward}, rear_ttc_reward:{rear_ttc_reward}")
        elif current_lane - last_lane == 1:
            # change left
            self.calculate_impact = -1
            center_front_dis = distance_to_front_vehicles[2]
            left_front_dis = distance_to_front_vehicles[1]
            dis=left_front_dis-center_front_dis
            reward=dis/self._env_config["vehicle_proximity"]*self._rl_configs["lane_change_reward"]
            # if left_front_dis > center_front_dis:
            #     reward = min((left_front_dis / center_front_dis - 1) * self.lane_change_reward, self.lane_change_reward)
            # else:
            #     reward = max((left_front_dis / center_front_dis - 1) * self.lane_change_reward, -self.lane_change_reward)
                # reward = 0
            ttc,rear_ttc_reward = self._ttc_reward(self.state["vehs"].center_rear_veh)
            LOG.reward_logger.debug(f"lane_change_reward:{reward}, rear_ttc_reward:{rear_ttc_reward}")

        return reward
