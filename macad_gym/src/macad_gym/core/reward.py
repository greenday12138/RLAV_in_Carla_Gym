import math, carla
import numpy as np
from macad_gym.core.utils.misc import get_speed, get_yaw_diff, get_sign, test_waypoint

class Reward(object):
    def __init__(self):
        self.reward = 0.0
        self.prev = None
        self.curr = None

    def compute_reward(self, prev_measurement, curr_measurement, flag):
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

    def destory(self):
        pass


class PDQNReward(Reward):
    def __init__(self):
        self.reward = 0.0
        self.prev = None
        self.curr = None
        
    def ttc_reward(self, ego_veh,target_veh,min_dis,TTC_THRESHOLD):
        """Caculate the time left before ego vehicle collide with target vehicle"""
        #TTC = float('inf')
        TTC=TTC_THRESHOLD
        if target_veh and ego_veh:
            distance = ego_veh.get_location().distance(target_veh.get_location())
            vehicle_len = max(abs(ego_veh.bounding_box.extent.x),
                                abs(ego_veh.bounding_box.extent.y)) + \
                            max(abs(target_veh.bounding_box.extent.x),
                                abs(target_veh.bounding_box.extent.y))
            distance -= vehicle_len
            # rel_speed = get_speed(ego_veh,False) - get_speed(target_veh, False)
            # if abs(rel_speed) > float(0.0000001):
            #     TTC = distance / rel_speed
            if distance < min_dis:
                TTC = 0.01
            else:
                distance -= min_dis
                rel_speed = get_speed(ego_veh,False) - get_speed(target_veh, False)
                if abs(rel_speed) > float(0.0000001):
                    TTC = distance / rel_speed
        # fTTC=-math.exp(-TTC)
        if TTC >= 0 and TTC <= TTC_THRESHOLD:
            fTTC = np.clip(np.log(TTC / TTC_THRESHOLD), -1, 0)
        else:
            fTTC = 0
            #TTC=TTC_THRESHOLD

        return TTC,fTTC
    
    def comfort_reward(self, fps, last_acc, acc, last_yaw, yaw):
        acc_jerk = -((acc - last_acc) * fps) ** 2 / ((6 * fps) ** 2)
        yaw_diff = math.degrees(get_yaw_diff(last_yaw, yaw))
        Yaw_jerk = -abs(yaw_diff) / 30
        return np.clip(acc_jerk * 0.5 + Yaw_jerk, -1, 0), yaw_diff
    
    def lane_center_reward(self, lane_center, ego_location):
        def compute(center,ego):
            Lcen=ego.distance(center.transform.location)
            center_yaw=lane_center.transform.get_forward_vector()
            dis=carla.Vector3D(ego.x-lane_center.transform.location.x,
                ego.y-lane_center.transform.location.y,0)
            Lcen*=get_sign(dis,center_yaw)
            return Lcen

        if not test_waypoint(lane_center, True):
            Lcen = 2.1
            fLcen = -2
            print('lane_center.lane_id, lane_center.road_id, flcen, lane_wid/2: ', lane_center.lane_id,
                    lane_center.road_id, fLcen, lane_center.lane_width / 2)
        else:
            Lcen =compute(lane_center,ego_location)
            fLcen = -abs(Lcen)/(lane_center.lane_width/2)
            # if self.current_action == Action.LANE_CHANGE_LEFT and self.current_lane == self.last_lane:
            #     # change left
            #     center_width=lane_center.lane_width
            #     lane_center=lane_center.get_left_lane()
            #     if lane_center is None:
            #         Lcen = 7
            #         fLcen = -2
            #     else:
            #         Lcen =compute(lane_center,ego_location)
            #         fLcen = -abs(Lcen) / (lane_center.lane_width/2+center_width)
            # elif self.current_action == Action.LANE_CHANGE_RIGHT and self.current_lane == self.last_lane:
            #     #change right
            #     center_width=lane_center.lane_width
            #     lane_center=lane_center.get_right_lane()
            #     if lane_center is None:
            #         Lcen = 7
            #         fLcen = -2
            #     else:
            #         Lcen =compute(lane_center,ego_location)
            #         fLcen=-abs(Lcen)/(lane_center.lane_width/2+center_width)
            # else:
            #     #lane follow and stop mode
            #     Lcen =compute(lane_center,ego_location)
            #     fLcen = -abs(Lcen)/(lane_center.lane_width/2)
            #print('pdqn_lane_center: Lcen, fLcen: ', Lcen, fLcen)
        return Lcen, fLcen

    def destory(self):
        pass