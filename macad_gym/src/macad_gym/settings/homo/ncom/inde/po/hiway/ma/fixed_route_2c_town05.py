#!/usr/bin/env python
import time
from macad_gym.envs.multi_env import MultiCarlaEnv
from macad_gym.envs.multi_env_pdqn import PDQNMultiCarlaEnv


class FixedRoute2CarTown05(MultiCarlaEnv):
    """A 3-way highway with traffic lights Multi-Agent Carla-Gym environment"""
    def __init__(self):
        self.configs = {
            "scenarios": "FR2C_TOWN5",
            "rl_parameters":{
                #Let the RL controller and PID controller alternatively take control every 20 episodes
                "switch_threshold": 10,
                #During pre-train steps, agent is only under PID control.
                "pre_train_steps": 640000,
                "train": True,
                "debug": False,
                "penalty": 40,
                "lane_change_reward": 40,
            },
            "env": {
                "server_map": "/Game/Carla/Maps/Town05_Opt",
                "render": False,
                "render_x_res": 2000,
                "render_y_res": 1500,
                "x_res": 500,
                "y_res": 500,
                "framestack": 1,
                "discrete_actions": False,
                "squash_action_logits": False,
                "verbose": True,
                "use_depth_camera": False,
                "send_measurements": True,
                "enable_planner": False,
                "spectator_loc": [140, 68, 9],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
                "fixed_route": True,
                "reward_policy": "SAC",
                #Distance for searching vehicles in front of ego vehicle, unit -- meters
                "vehicle_proximity": 50.0, 
                #Distance for searching traffic light in front of ego vehicle, unit -- meters, attention: this value is tricky
                "traffic_light_proximity": 50.0,  
                #Min distance between two vehicles, unit -- meters
                "min_distance": 15.0,
                #Activate hybrid mode for Traffic Manager
                "hybrid": True,
                "ignore_traffic_light": False,
                #Set lane change behaviors of Traffic Manager
                "auto_lane_change": False, 
                #Distance between generated waypoints
                "sampling_resolution": 4.0,
                #The number of look-ahead waypoints in each step
                "buffer_size": 50,
            },
            "actors": {
                "car1": {
                    "type": "vehicle_4W",
                    "blueprint": "vehicle.tesla.model3",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "scenarios": "FR2C_TOWN5_CAR1",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": True,
                    "render": True,
                    "x_res": 500,
                    "y_res": 500,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    #Speed limit for hero vehicle, km/h
                    "speed_limit": 90.0,
                    #Speed threshold for hero vehicle, running in start phase before speed reach such threshold, km/h
                    "speed_threshold": 20.0,
                    #If hero vehicle speed reaches below this threshold across multiple steps, truncated this episode prematurely, km/h
                    "speed_min": 0.36,
                    #Steer bound for hero vehicle controller
                    "steer_bound": 1.0,
                    #Throttle bound for ego vehicle controller
                    "throttle_bound": 1.0,
                    #Brake bound for ego vehicle controller
                    "brake_bound": 1.0,
                },
                "car2": {
                    "type": "vehicle_4W",
                    "blueprint": "vehicle.tesla.model3",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "scenarios": "FR2C_TOWN5_CAR2",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": True,
                    "render": True,
                    "x_res": 500,
                    "y_res": 500,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    #Speed limit for hero vehicle, km/h
                    "speed_limit": 90.0,
                    #Speed threshold for hero vehicle, running in start phase before speed reach such threshold, km/h
                    "speed_threshold": 20.0,
                    #If hero vehicle speed reaches below this threshold across multiple steps, truncated this episode prematurely, km/h
                    "speed_min": 0.36,
                    #Steer bound for hero vehicle controller
                    "steer_bound": 1.0,
                    #Throttle bound for ego vehicle controller
                    "throttle_bound": 1.0,
                    #Brake bound for ego vehicle controller
                    "brake_bound": 1.0,
                },
                "car3": {
                    "type": "vehicle_4W",
                    "blueprint": "vehicle.tesla.model3",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "scenarios": "FR2C_TOWN5_CAR2",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": True,
                    "render": True,
                    "x_res": 500,
                    "y_res": 500,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    #Speed limit for hero vehicle, km/h
                    "speed_limit": 90.0,
                    #Speed threshold for hero vehicle, running in start phase before speed reach such threshold, km/h
                    "speed_threshold": 20.0,
                    #If hero vehicle speed reaches below this threshold across multiple steps, truncated this episode prematurely, km/h
                    "speed_min": 0.36,
                    #Steer bound for hero vehicle controller
                    "steer_bound": 1.0,
                    #Throttle bound for ego vehicle controller
                    "throttle_bound": 1.0,
                    #Brake bound for ego vehicle controller
                    "brake_bound": 1.0,
                }
            },
        }
        super(FixedRoute2CarTown05, self).__init__(self.configs)


class PDQNFixedRoute2CarTown05(PDQNMultiCarlaEnv):
    """A 3-way highway with traffic lights Multi-Agent Carla-Gym environment, specilized for pdqn"""
    def __init__(self):
        self.configs = {
            "scenarios": "FR2C_TOWN5",
            "rl_parameters":{
                #Let the RL controller and PID controller alternatively take control every 20 episodes
                "switch_threshold": 10,
                #During pre-train steps, agent is only under PID control.
                "pre_train_steps": 640000,
                "train": True,
                "debug": False,
                "modify_steer": True,
                "penalty": 40,
                "lane_penalty": 20,
                "lane_change_reward": 40,
            },
            "env": {
                "server_map": "/Game/Carla/Maps/Town05_Opt",
                "render": False,
                "render_x_res": 2000,
                "render_y_res": 1500,
                "x_res": 500,
                "y_res": 500,
                "framestack": 1,
                "discrete_actions": False,
                "squash_action_logits": False,
                "verbose": True,
                "use_depth_camera": False,
                "send_measurements": True,
                "enable_planner": False,
                "spectator_loc": [140, 68, 9],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
                "fixed_route": True,
                "reward_policy": "PDQN",
                #Distance for searching vehicles in front of ego vehicle, unit -- meters
                "vehicle_proximity": 50.0, 
                #Distance for searching traffic light in front of ego vehicle, unit -- meters, attention: this value is tricky
                "traffic_light_proximity": 50.0,  
                #Min distance between two vehicles, unit -- meters
                "min_distance": 15.0,
                #Activate hybrid mode for Traffic Manager
                "hybrid": True,
                "ignore_traffic_light": False,
                #Set lane change behaviors of Traffic Manager
                "auto_lane_change": False, 
                #Distance between generated waypoints
                "sampling_resolution": 4.0,
                #The number of look-ahead waypoints in each step
                "buffer_size": 50,
            },
            "actors": {
                "car1": {
                    "type": "vehicle_4W",
                    "blueprint": "vehicle.tesla.model3",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "scenarios": "FR2C_TOWN5_CAR1",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": True,
                    "render": True,
                    "x_res": 500,
                    "y_res": 500,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    #Speed limit for hero vehicle, km/h
                    "speed_limit": 90.0,
                    #Speed threshold for hero vehicle, running in start phase before speed reach such threshold, km/h
                    "speed_threshold": 20.0,
                    #If hero vehicle speed reaches below this threshold across multiple steps, truncated this episode prematurely, km/h
                    "speed_min": 0.36,
                    #Steer bound for hero vehicle controller
                    "steer_bound": 1.0,
                    #Throttle bound for ego vehicle controller
                    "throttle_bound": 1.0,
                    #Brake bound for ego vehicle controller
                    "brake_bound": 1.0,
                },
                "car2": {
                    "type": "vehicle_4W",
                    "blueprint": "vehicle.tesla.model3",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "scenarios": "FR2C_TOWN5_CAR2",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": True,
                    "render": True,
                    "x_res": 500,
                    "y_res": 500,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    #Speed limit for hero vehicle, km/h
                    "speed_limit": 90.0,
                    #Speed threshold for hero vehicle, running in start phase before speed reach such threshold, km/h
                    "speed_threshold": 20.0,
                    #If hero vehicle speed reaches below this threshold across multiple steps, truncated this episode prematurely, km/h
                    "speed_min": 0.36,
                    #Steer bound for hero vehicle controller
                    "steer_bound": 1.0,
                    #Throttle bound for ego vehicle controller
                    "throttle_bound": 1.0,
                    #Brake bound for ego vehicle controller
                    "brake_bound": 1.0,
                },
                "car3": {
                    "type": "vehicle_4W",
                    "blueprint": "vehicle.tesla.model3",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "scenarios": "FR2C_TOWN5_CAR2",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": True,
                    "render": True,
                    "x_res": 500,
                    "y_res": 500,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    #Speed limit for hero vehicle, km/h
                    "speed_limit": 90.0,
                    #Speed threshold for hero vehicle, running in start phase before speed reach such threshold, km/h
                    "speed_threshold": 20.0,
                    #If hero vehicle speed reaches below this threshold across multiple steps, truncated this episode prematurely, km/h
                    "speed_min": 0.36,
                    #Steer bound for hero vehicle controller
                    "steer_bound": 1.0,
                    #Throttle bound for ego vehicle controller
                    "throttle_bound": 1.0,
                    #Brake bound for ego vehicle controller
                    "brake_bound": 1.0,
                }
            },
        }
        super(PDQNFixedRoute2CarTown05, self).__init__(self.configs)

if __name__ == "__main__":
    env = FixedRoute2CarTown05()
    configs = env.configs
    for ep in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = configs["env"]
        actor_configs = configs["actors"]
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
                action_dict[actor_id] = 3  # Forward
            else:
                action_dict[actor_id] = [1, 0]  # test values

        start = time.time()
        i = 0
        done = {"__all__": False}
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            obs, reward, done, info = env.step(action_dict)
            # action_dict = get_next_actions(info, env.discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew",
                                  "done{}"]).format(i, reward,
                                                    total_reward_dict, done))

            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))
