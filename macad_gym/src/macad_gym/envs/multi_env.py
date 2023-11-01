"""
multi_env.py: Multi-actor environment interface for CARLA-Gym
Should support two modes of operation. See CARLA-Gym developer guide for
more information
__author__: @Praveen-Palanisamy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import weakref
import gc
import argparse
import atexit
import json
import os
import random
import signal
import time
import traceback
import math
import pygame
import carla
import numpy as np  # linalg.norm is used

from collections import deque
from gym.spaces import Box, Discrete, Tuple, Dict

from macad_gym import LOG_PATH, RETRIES_ON_ERROR
from macad_gym.viz.logger import LOG
from macad_gym.core.utils.state import StateDAO
from macad_gym.core.controllers.traffic import apply_traffic, hero_autopilot
from macad_gym.multi_actor_env import MultiActorEnv
from macad_gym.core.maps.nodeid_coord_map import MAP_TO_COORDS_MAPPING
from macad_gym.core.utils.misc import (get_lane_center, get_yaw_diff, test_waypoint, 
                                       is_within_distance_ahead, get_projection, draw_waypoints,
                                       get_speed, preprocess_image)
from macad_gym.core.simulator.carla_provider import CarlaConnector, CarlaError, CarlaDataProvider, termination_cleanup
from macad_gym.core.utils.wrapper import (ROAD_OPTION_TO_COMMANDS_MAPPING, DISTANCE_TO_GOAL_THRESHOLD, 
                                          ORIENTATION_TO_GOAL_THRESHOLD, DISCRETE_ACTIONS, WEATHERS, 
                                          get_next_actions, DEFAULT_MULTIENV_CONFIG, print_measurements, 
                                          Truncated, SpeedState, ControlInfo)

# from macad_gym.core.sensors.utils import get_transform_from_nearest_way_point
from macad_gym.core.utils.reward import Reward, PDQNReward, SACReward
from macad_gym.core.sensors.hud import HUD
from macad_gym.viz.render import Render
from macad_gym.core.scenarios import Scenarios

# The following imports require carla to be imported already.
from macad_gym.core.sensors.camera_manager import CameraManager, CAMERA_TYPES
from macad_gym.core.sensors.derived_sensors import LaneInvasionSensor, CollisionSensor
from macad_gym.core.controllers.keyboard_control import KeyboardControl
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner_dao import (  # noqa: E501
    GlobalRoutePlannerDAO)
from macad_gym.core.controllers.route_planner import RoutePlanner
from macad_gym.core.controllers.local_planner import LocalPlanner
from macad_gym.core.controllers.basic_agent import Basic_Agent

from macad_gym.core.maps.nav_utils import PathTracker  # noqa: E402
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner import (  # noqa: E402, E501
    GlobalRoutePlanner)

signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(CarlaConnector.clean_up)

MultiAgentEnvBases = [MultiActorEnv]
try:
    from ray.rllib.env import MultiAgentEnv

    MultiAgentEnvBases.append(MultiAgentEnv)
except ImportError:
    LOG.multi_env_logger.warning("\n Disabling RLlib support.", exc_info=True)



class MultiCarlaEnv(*MultiAgentEnvBases):
    def __init__(self, configs=None):
        """MACAD-Gym environment implementation.

        Provides a generic MACAD-Gym environment implementation that can be
        customized further to create new or variations of existing
        multi-agent learning environments. The environment settings, scenarios
        and the actors in the environment can all be configured using
        the `configs` dict.

        Args:
            configs (dict): Configuration for environment specified under the
                `env` key and configurations for each actor specified as dict
                under `actor`.
                Example:
                    >>> configs = {"env":{
                    "server_map":"/Game/Carla/Maps/Town05",
                    "discrete_actions":True,...},
                    "actor":{
                    "actor_id1":{"enable_planner":True,...},
                    "actor_id2":{"enable_planner":False,...}
                    }}
        """

        if configs is None:
            configs = DEFAULT_MULTIENV_CONFIG
        # Functionalities classes
        configs["scenarios"] = Scenarios.resolve_scenarios_parameter(
            configs["scenarios"]
        )

        self._scenario_config = configs["scenarios"]
        self._env_config = configs["env"]
        self._actor_configs = configs["actors"]
        self._rl_configs = configs["rl_parameters"]
        if self._env_config["reward_policy"] == "PDQN":
            self._reward_policy = PDQNReward(configs)
        elif self._env_config["reward_policy"] == "SAC":
            self._reward_policy = SACReward(configs)
        else:
            self._reward_policy = Reward(configs)

        # At most one actor can be manual controlled
        manual_control_count = 0
        for _, actor_config in self._actor_configs.items():
            if actor_config["manual_control"]:
                if "vehicle" not in actor_config["type"]:
                    raise ValueError("Only vehicles can be manual controlled.")

                manual_control_count += 1

        assert manual_control_count <= 1, (
            "At most one actor can be manually controlled. "
            f"Found {manual_control_count} actors with manual_control=True"
        )

        # Camera position is problematic for certain vehicles and even in
        # autopilot they are prone to error
        self.exclude_hard_vehicles = False
        # list of str: Supported values for `type` filed in `actor_configs`
        # for actors than can be actively controlled
        self._supported_active_actor_types = [
            "vehicle_4W",
            "vehicle_2W",
            "pedestrian",
            "traffic_light",
        ]
        # list of str: Supported values for `type` field in `actor_configs`
        # for actors that are passive. Example: A camera mounted on a pole
        self._supported_passive_actor_types = ["camera"]

        # Set attributes as in gym's specs
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": "human"}

        # Belongs to env_config.
        self._server_map = self._env_config["server_map"].split("/")[-1]
        self._render = self._env_config["render"]
        self._framestack = self._env_config["framestack"]
        self._discrete_actions = self._env_config["discrete_actions"]
        self._squash_action_logits = self._env_config["squash_action_logits"]
        self._verbose = self._env_config["verbose"]
        self._render_x_res = self._env_config["render_x_res"]
        self._render_y_res = self._env_config["render_y_res"]
        self._x_res = self._env_config["x_res"]
        self._y_res = self._env_config["y_res"]
        self._use_depth_camera = self._env_config["use_depth_camera"]
        self._sync_server = self._env_config["sync_server"]
        self._fixed_delta_seconds = self._env_config["fixed_delta_seconds"]

        # Initialize to be compatible with cam_manager to set HUD.
        pygame.font.init()  # for HUD
        self._hud = HUD(self._render_x_res, self._render_y_res)

        # For manual_control
        self._control_clock = None
        self._manual_controller = None
        self._manual_control_camera_manager = None

        # Render related
        Render.resize_screen(self._render_x_res, self._render_y_res)

        self._camera_poses, window_dim = Render.get_surface_poses(
            [self._x_res, self._y_res], self._actor_configs
        )

        if manual_control_count == 0:
            Render.resize_screen(window_dim[0], window_dim[1])
        else:
            self._manual_control_render_pose = (0, window_dim[1])
            Render.resize_screen(
                max(self._render_x_res, window_dim[0]),
                self._render_y_res + window_dim[1],
            )

        # Actions space
        if self._discrete_actions:
            self.action_space = Dict(
                {
                    actor_id: Discrete(len(DISCRETE_ACTIONS))
                    for actor_id in self._actor_configs.keys()
                }
            )
        else:
            self.action_space = Dict(
                {
                    actor_id: Box(-1.0, 1.0, shape=(2,))
                    for actor_id in self._actor_configs.keys()
                }
            )

        # Output space of images after preprocessing
        if self._use_depth_camera:
            self._image_space = Box(
                0.0, 255.0, shape=(self._y_res, self._x_res, 1 * self._framestack)
            )
        else:
            self._image_space = Box(
                -1.0, 1.0, shape=(self._y_res, self._x_res, 3 * self._framestack)
            )

        # Observation space in output
        if self._env_config["send_measurements"]:
            self.observation_space = Dict(
                {
                    actor_id: Tuple(
                        [
                            self._image_space,
                            Dict({
                                "left_waypoints": Box(-2.0, 2.0, shape=(10, 3), dtype=np.float32), 
                                "center_waypoints": Box(-2.0, 2.0, shape=(10, 3), dtype=np.float32),
                                "right_waypoints": Box(-2.0, 2.0, shape=(10, 3), dtype=np.float32), 
                                "vehicle_info": Box(-np.inf, np.inf,shape=(6, 4), dtype=np.float32),
                                "hero_vehicle": Box(-2.0, 2.0, shape=(1, 6), dtype=np.float32),
                                "light":Box(-np.inf, np.inf, shape=(1, 3), dtype=np.float32),
                            })
                        ]
                    )
                    for actor_id in self._actor_configs.keys()
                }
            )
        else:
            self.observation_space = Dict(
                {actor_id: self._image_space for actor_id in self._actor_configs.keys()}
            )

        # Following info will only be initialized once in _init_server() (env configs)
        # Set appropriate node-id to coordinate mappings for Town01 or Town02.
        self.pos_coor_map = MAP_TO_COORDS_MAPPING[self._server_map]
        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._carla = None
        self.SWITCH_THRESHOLD = self._rl_configs["switch_threshold"]
        if self._rl_configs["train"]:
            self.pre_train_steps = self._rl_configs["pre_train_steps"]
        else:
            self.pre_train_steps = 0
        self._debug = self._rl_configs["debug"]
        self._npc_vehicles_spawn_points = []
        self._agents = {}  # Dictionary of macad_agents with agent_id as key
        # Env statistics
        self._previous_rewards = {}
        self._total_reward = {}
        self._rl_switch = False    # rl_switch: True--currently RL in control, False--currently PID in control
        self._switch_count=0
        self._num_episodes = {}
        self._total_steps = 0
        self._rl_control_steps = 0

        # Following info will be reset during reset phase
        self._weather = None
        self._scenario_map = {}  # Dictionary with current scenario map config
        self._start_pos = {}  # Start pose for each actor
        self._end_pos = {}  # End pose for each actor
        self._start_coord = {}
        self._end_coord = {}
        self._measurements_file_dict = {}
        self._time_steps = {}
        self._vel_buffer = {}   # Dictionary recording hero vehicles' velocity
        self._cameras = {}  # Dictionary of sensors with actor_id as key
        self._actors = {}  # Dictionary of actors with actor_id as key
        self._npc_vehicles = []  # List of NPC vehicles
        self._npc_pedestrians = []  # List of NPC pedestrians
        self._path_trackers = {}  # Dictionary of sensors with actor_id as key
        self._collisions = {}  # Dictionary of sensors with actor_id as key
        self._lane_invasions = {}  # Dictionary of sensors with actor_id as key
        self._speed_state = {}  # Dictionary of actors' running state with actor_id as key
        self._state_getter = None
        self._auto_controller = {} # Dictionary of vehicles' automatic controllers

        # Following info will be modified every step
        self._prev_measurement = {}
        self._cur_measurement = {}
        self._prev_image = {}
        self._obs_dict = {}
        self._done_dict = {}
        self._truncated_dict = {}
        self._state = {
            "wps":{},   #Dictionary of waypoints info with actor_id as keyw
            "lights":{}, #Dictionary of traffic lights info with actor_id as key
            "vehs":{},  #Dictionary of other vehicles info with actor_id as key
        }
        # Lane change info and action
        self.control_info = {}
        self.last_lane,self.current_lane = {}, {}
        self.last_action,self.current_action = {}, {}
        self.last_target_lane,self.current_target_lane = {}, {}
        self.last_light_state = {}
        self.last_acc = {}  # hero vehicle acceration along s in last step
        self.last_yaw = {}

    def _init_server(self):
        """Initialize carla server and client

        Returns:
            N/A
        """
        LOG.multi_env_logger.info("Initializing new Carla server...")
        self._carla = CarlaConnector(self._server_map, self._env_config)

        # Set the spectator/server view if rendering is enabled
        if self._render and self._env_config.get("spectator_loc"):
            spectator = self._carla.get_spectator(LOG.multi_env_logger)
            spectator_loc = carla.Location(*self._env_config["spectator_loc"])
            d = 6.4
            angle = 160  # degrees
            a = math.radians(angle)
            location = (
                carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + spectator_loc)
            spectator.set_transform(
                carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15)))

        if self._env_config.get("enable_planner"):
            planner_dao = GlobalRoutePlannerDAO(self.map)
            self.planner = GlobalRoutePlanner(planner_dao)
            self.planner.setup()

        if self._env_config.get("fixed_route"):
            self._npc_vehicles_spawn_points = \
                RoutePlanner(weakref.proxy(self._carla._map), sampling_resolution=4.0).get_spawn_points()
            
        Render.init()

    def _clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A
        """
        try:
            [colli.destroy() for colli in self._collisions.values()]
            [lane.destroy() for lane in self._lane_invasions.values()]
            [camera.destroy() for camera in self._cameras.values()]
            for npc in self._npc_vehicles:
                if npc.is_alive:
                    npc.destroy()
            for actor in self._actors.values():
                if actor.is_alive:
                    actor.destroy()
            for npc in zip(*self._npc_pedestrians):
                npc[1].stop()  # stop controller
                npc[0].destroy()  # kill entity
            self._carla.tick(LOG.multi_env_logger)
        except RuntimeError as e:
            raise CarlaError(e.args)
        # Note: the destroy process for cameras is handled in camera_manager.py

        self._vel_buffer = {}
        self._cameras = {}
        self._actors = {}
        self._npc_vehicles = []
        self._npc_pedestrians = []
        self._path_trackers = {}
        self._collisions = {}
        self._lane_invasions = {}
        self._speed_state = {} 
        del self._state_getter 
        self._state_getter = None
        del self._auto_controller
        self._auto_controller = {} 

        LOG.multi_env_logger.info("Cleaned-up the world...")

    def _clear_server_state(self):
        """Clear server process"""
        LOG.multi_env_logger.info("Clearing Carla server state")
        try:
            if self._carla:
                self._carla.disconnect()
                del self._carla
                self._carla = None
                Render.quit()
        except Exception as e:
            LOG.multi_env_logger.exception("Error disconnecting client: {}".format(e))

        gc.collect()

    def reset(self, seed=None, options=None):
        """Reset the carla world, call _init_server()

        Returns:
            N/A
        """
        # World reset and new scenario selection if multiple are available
        self._load_scenario(self._scenario_config)

        for retry in range(RETRIES_ON_ERROR):
            try:
                if not self._carla:
                    self._init_server()
                    self._reset(clean_world=False)
                else:
                    self._reset()
                break
            except CarlaError as e:
                LOG.multi_env_logger.exception(e.args)
                self._clear_server_state()
                continue
            except AttributeError as e:
                LOG.multi_env_logger.exception(e.args)
                if e.args[0].find("'NoneType' object has no attribute") == -1:
                    raise e
                else:
                    self._clear_server_state()
                    continue
            except Exception as e:
                LOG.multi_env_logger.exception("Error during reset: {}".format(traceback.format_exc()))
                LOG.multi_env_logger.error("reset(): Retry #: {}/{}".format(retry + 1, RETRIES_ON_ERROR))
                self._clear_server_state()
                raise e
            
        # vehicle controller switch
        if not self._debug:
            if self._total_steps-self._rl_control_steps <self.pre_train_steps:
                #During pre-train steps, let rl and pid alternatively take control
                if self._rl_switch:
                    if self._switch_count>=self.SWITCH_THRESHOLD:
                        self._rl_switch=False
                        self._switch_count=0
                    else:
                        self._switch_count+=1
                else:
                    self._rl_switch=True
                    self._switch_count+=1
            else:
                self._rl_switch=True
        else:
            self._rl_switch=False
            # self.sim_world.debug.draw_point(self.ego_spawn_point.location,size=0.3,life_time=0)
            # while (True):
            #     spawn_point=random.choice(self.spawn_points).location
            #     if self.map.get_waypoint(spawn_point).lane_id==self.map.get_waypoint(self.ego_spawn_point.location).lane_id:
            #         break
            # self.controller.set_destination(spawn_point)     

        # Set appropriate initial values for all actors
        for actor_id, actor_config in self._actor_configs.items():
            cam = self._cameras[actor_id]
            # Wait for the sensor (camera) actor to start streaming
            # Shouldn't take too long
            while cam.callback_count == 0:
                if self._sync_server:
                    self._carla.tick(LOG.multi_env_logger)
                    # `wait_for_tick` is no longer needed, see https://github.com/carla-simulator/carla/pull/1803
                    # self.world.wait_for_tick()
            if cam.image is None:
                LOG.multi_env_logger.debug(f"callback_count:{actor_id}:{cam.callback_count}")
            # Actor correctly reset
            self._done_dict[actor_id] = False
            self._truncated_dict[actor_id] = Truncated.FALSE
            self._total_reward[actor_id] = None
            
            #update prev step info
            current_lane=get_lane_center(weakref.proxy(self._carla._map), self._actors[actor_id].get_location()).lane_id
            self.last_lane[actor_id] = self.current_lane[actor_id] = current_lane
            self.last_light_state[actor_id] = None
            self.last_acc[actor_id] = 0
            self.last_yaw[actor_id] = carla.Vector3D()
            self.control_info[actor_id] = ControlInfo()

            #update time steps and prev step info
            current_lane=get_lane_center(weakref.proxy(self._carla._map), self._actors[actor_id].get_location()).lane_id
            if actor_id in self._num_episodes.keys():
                self._num_episodes[actor_id]+=1
            else:
                self._num_episodes[actor_id]=0
            self._time_steps[actor_id] = 0

            state = self._read_observation(actor_id)
            obs = self._encode_obs(actor_id, cam.image, state)
            self._obs_dict[actor_id] = obs
            self._prev_measurement[actor_id] = self._cur_measurement[actor_id]
            self._prev_image[actor_id] = None

        return self._obs_dict, {}

    def _spawn_new_actor(self, actor_id):
        """Spawn an agent as per the blueprint at the given pose

        Args:
            blueprint: Blueprint of the actor. Can be a Vehicle or Pedestrian
            pose: carla.Transform object with location and rotation

        Returns:
            An instance of a subclass of carla.Actor. carla.Vehicle in the case
            of a Vehicle agent.

        """
        actor_type = self._actor_configs[actor_id].get("type", "vehicle_4W")
        if actor_type not in self._supported_active_actor_types:
            LOG.multi_env_logger.error(f"Unsupported actor type:{actor_type}. Using vehicle_4W as the type")
            actor_type = "vehicle_4W"

        if actor_type == "traffic_light":
            # Traffic lights already exist in the world & can't be spawned.
            # Find closest traffic light actor in world.actor_list and return
            from macad_gym.core.controllers import traffic_lights

            loc = carla.Location(
                self._start_pos[actor_id][0],
                self._start_pos[actor_id][1],
                self._start_pos[actor_id][2],
            )
            rot = (
                self._carla._map.get_waypoint(loc, project_to_road=True).transform.rotation)
            #: If yaw is provided in addition to (X, Y, Z), set yaw
            if len(self._start_pos[actor_id]) > 3:
                rot.yaw = self._start_pos[actor_id][3]
            transform = carla.Transform(loc, rot)
            self._actor_configs[actor_id]["start_transform"] = transform
            tls = traffic_lights.get_tls(weakref.proxy(self._carla._world), transform, sort=True)
            return tls[0][0]  #: Return the key (carla.TrafficLight object) of
            #: closest match

        if actor_type == "pedestrian":
            blueprints = self._carla.get_blueprint_library(LOG.multi_env_logger).filter(
                "walker.pedestrian.*"
            )

        elif actor_type == "vehicle_4W":
            blueprints = self._carla.get_blueprint_library(LOG.multi_env_logger).filter("vehicle")
            # Further filter down to 4-wheeled vehicles
            blueprints = [
                b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 4
            ]
            if self.exclude_hard_vehicles:
                blueprints = list(
                    filter(
                        lambda x: not (
                            x.id.endswith("microlino")
                            or x.id.endswith("carlacola")
                            or x.id.endswith("cybertruck")
                            or x.id.endswith("t2")
                            or x.id.endswith("sprinter")
                            or x.id.endswith("firetruck")
                            or x.id.endswith("ambulance")
                        ),
                        blueprints,
                    )
                )
        elif actor_type == "vehicle_2W":
            blueprints = self._carla.get_blueprint_library(LOG.multi_env_logger).filter("vehicle")
            # Further filter down to 2-wheeled vehicles
            blueprints = [
                b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 2
            ]

        if self._actor_configs[actor_id]["blueprint"]:
            for blue in blueprints:
                if blue.id == self._actor_configs[actor_id]["blueprint"]:
                    blueprint = blue
                    break
        else:
            blueprint = random.choice(blueprints)
        blueprint.set_attribute('role_name', 'hero')
        loc = carla.Location(
            x=self._start_pos[actor_id][0],
            y=self._start_pos[actor_id][1],
            z=self._start_pos[actor_id][2]+0.1,
        )
        rot = (
            self._carla._map.get_waypoint(loc, project_to_road=True).transform.rotation)
        #: If yaw is provided in addition to (X, Y, Z), set yaw
        if len(self._start_pos[actor_id]) > 3:
            rot.yaw = self._start_pos[actor_id][3]
        transform = carla.Transform(loc, rot)
        #draw_waypoints(self.world, [self.map.get_waypoint(transform.location)], life_time=0)
        self._actor_configs[actor_id]["start_transform"] = transform
        vehicle = None
        for retry in range(RETRIES_ON_ERROR):
            vehicle = self._carla._world.try_spawn_actor(blueprint, transform)
            if self._sync_server:
                self._carla.tick(LOG.multi_env_logger)
            if vehicle is not None and vehicle.get_location().z > 0.0:
                # Register it under traffic manager
                # Walker vehicle type does not have autopilot. Use walker controller ai
                if actor_type == "pedestrian":
                    # vehicle.set_simulate_physics(False)
                    pass
                break
            # Wait to see if spawn area gets cleared before retrying
            # time.sleep(0.5)
            # self._clean_world()
            LOG.multi_env_logger.error("spawn_actor: Retry#:{}/{}".format(retry + 1, RETRIES_ON_ERROR))
        if vehicle is None:
            # Request a spawn one last time possibly raising the error
            vehicle = self._carla._world.try_spawn_actor(blueprint, transform)
        return vehicle

    def _reset(self, clean_world=True):
        """Reset the state of the actors.
        A "soft" reset is performed in which the existing actors are destroyed
        and the necessary actors are spawned into the environment without
        affecting other aspects of the environment.
        If the "soft" reset fails, a "hard" reset is performed in which
        the environment's entire state is destroyed and a fresh instance of
        the server is created from scratch. Note that the "hard" reset is
        expected to take more time. In both of the reset modes ,the state/
        pose and configuration of all actors (including the sensor actors) are
        (re)initialized as per the actor configuration.

        Returns:
            dict: Dictionaries of observations for actors.

        Raises:
            RuntimeError: If spawning an actor at its initial state as per its'
            configuration fails (eg.: Due to collision with an existing object
            on that spot). This Error will be handled by the caller
            `self.reset()` which will perform a "hard" reset by creating
            a new server instance
        """
        if clean_world:
            self._clean_world()
            # set new log file
            LOG.set_log(os.path.join(LOG_PATH, str(self._num_episodes[
                list(self._num_episodes.keys())[0]
            ])))
            
        weather_num = 0
        if "weather_distribution" in self._scenario_map:
            weather_num = random.choice(self._scenario_map["weather_distribution"])
            if weather_num not in WEATHERS:
                weather_num = 0

        self._carla.set_weather(WEATHERS[weather_num], LOG.multi_env_logger)
        weas = self._carla.get_weather(LOG.multi_env_logger)
        self._weather = [weas.cloudiness, weas.precipitation, weas.precipitation_deposits, weas.wind_intensity]

        for actor_id, actor_config in self._actor_configs.items():
            if self._done_dict.get("__all__", True) or \
                    self._truncated_dict.get("__all__", Truncated.FALSE)!=Truncated.FALSE:
                self._measurements_file_dict[actor_id] = None
                actor_config = self._actor_configs[actor_id]

                # Try to spawn actor (soft reset) or fail and reinitialize the server before get back here
                try:
                    self._actors[actor_id] = self._spawn_new_actor(actor_id)
                    self._auto_controller[actor_id] = Basic_Agent(self._actors[actor_id], dt=self._fixed_delta_seconds,
                        opt_dict={'ignore_traffic_lights': self._env_config["ignore_traffic_light"],
                                  'ignore_stop_signs': True, 
                                  'sampling_resolution': self._env_config["sampling_resolution"],
                                  'max_steering': self._actor_configs[actor_id]["steer_bound"], 
                                  'max_throttle': self._actor_configs[actor_id]["throttle_bound"],
                                  'max_brake': self._actor_configs[actor_id]["brake_bound"], 
                                  'buffer_size': self._env_config["buffer_size"], 
                                  'target_speed':self._actor_configs[actor_id]["speed_limit"],
                                  'ignore_front_vehicle':False,
                                  'ignore_change_gap':False,
                                #   'ignore_front_vehicle': random.choice([False,True]),
                                #   'ignore_change_gap': random.choice([True, True, False]), 
                                  'lanechanging_fps': random.choice([40, 50, 60]),
                                  'random_lane_change':True})
                    self._speed_state[actor_id] = SpeedState.START
                except RuntimeError as spawn_err:
                    del self._done_dict[actor_id]
                    del self._truncated_dict[actor_id]
                    # Chain the exception & re-raise to be handled by the caller `self.reset()`
                    raise spawn_err from RuntimeError(
                        "Unable to spawn actor:{}".format(actor_id))

                if self._env_config["enable_planner"]:
                    self._path_trackers[actor_id] = PathTracker(
                        weakref.proxy(self._carla._world),
                        self.planner,
                        (
                            self._start_pos[actor_id][0],
                            self._start_pos[actor_id][1],
                            self._start_pos[actor_id][2],
                        ),
                        (
                            self._end_pos[actor_id][0],
                            self._end_pos[actor_id][1],
                            self._end_pos[actor_id][2],
                        ),
                        self._actors[actor_id],
                    )

                # Spawn collision and lane sensors if necessary
                if actor_config["collision_sensor"] == "on":
                    collision_sensor = CollisionSensor(self._actors[actor_id])
                    self._collisions.update({actor_id: collision_sensor})
                if actor_config["lane_sensor"] == "on":
                    lane_sensor = LaneInvasionSensor(self._actors[actor_id])
                    self._lane_invasions.update({actor_id: lane_sensor})

                # Spawn cameras
                pygame.font.init()  # for HUD
                hud = HUD(self._env_config["x_res"], self._env_config["y_res"])
                camera_manager = CameraManager(self._actors[actor_id], hud)
                if actor_config["log_images"]:
                    # TODO: The recording option should be part of config
                    # 1: Save to disk during runtime
                    # 2: save to memory first, dump to disk on exit
                    camera_manager.set_recording_option(1)

                # in CameraManger's._sensors
                camera_type = self._actor_configs[actor_id]["camera_type"]
                camera_pos = getattr(
                    self._actor_configs[actor_id], "camera_position", 2
                )
                camera_types = [ct.name for ct in CAMERA_TYPES]
                assert (
                    camera_type in camera_types
                ), "Camera type `{}` not available. Choose in {}.".format(
                    camera_type, camera_types
                )
                camera_manager.set_sensor(
                    CAMERA_TYPES[camera_type].value - 1, int(camera_pos), 
                    notify=False, force_respawn=True
                )
                assert camera_manager.sensor.is_listening
                self._cameras.update({actor_id: camera_manager})

                # Spawn vehicle velocity recorder
                # the maxlen is related to traffic_light.get_green_time()
                self._vel_buffer.update({actor_id: deque(maxlen=10)})

                # Manual Control
                if actor_config["manual_control"]:
                    self._control_clock = pygame.time.Clock()

                    self._manual_controller = KeyboardControl(
                        self, actor_config["auto_control"]
                    )
                    self._manual_controller.actor_id = actor_id

                    self.world.on_tick(self._hud.on_world_tick)
                    self._manual_control_camera_manager = CameraManager(
                        self._actors[actor_id], self._hud
                    )
                    self._manual_control_camera_manager.set_sensor(
                        CAMERA_TYPES["rgb"].value - 1, pos=2, notify=False
                    )

                self._start_coord.update({
                        actor_id: [
                            self._start_pos[actor_id][0] // 100,
                            self._start_pos[actor_id][1] // 100,
                        ]
                    }
                )
                self._end_coord.update({
                        actor_id: [
                            self._end_pos[actor_id][0] // 100,
                            self._end_pos[actor_id][1] // 100,
                        ]
                    }
                )

                LOG.multi_env_logger.info(
                    "Actor: {} start_pos_xyz(coordID): {} ({}), "
                    "end_pos_xyz(coordID) {} ({})".format(
                        actor_id,
                        self._start_pos[actor_id],
                        self._start_coord[actor_id],
                        self._end_pos[actor_id],
                        self._end_coord[actor_id],
                    )
                )
        self._done_dict["__all__"] = False
        self._truncated_dict["__all__"] = Truncated.FALSE
        LOG.multi_env_logger.info("New episode initialized with actors:{}".format(self._actors.keys()))

        self._state_getter = StateDAO({
            "scenario_config": self._scenario_config,
            "env_config": self._env_config,
            "actor_config": self._actor_configs,
            "rl_config": self._rl_configs,
            "actors": self._actors,
            "world": weakref.proxy(self._carla._world),
            "map": weakref.proxy(self._carla._map),
        })

        self._npc_vehicles, self._npc_pedestrians = apply_traffic(
            weakref.proxy(self._carla._world),
            weakref.proxy(self._carla._traffic_manager),
            self._env_config,
            self._scenario_map.get("num_vehicles", 0),
            self._scenario_map.get("num_pedestrians", 0),
            safe = True,
            route_points = self._npc_vehicles_spawn_points
        )

    def _load_scenario(self, scenario_parameter):
        self._scenario_map = {}
        # If config contains a single scenario, then use it,
        # if it's an array of scenarios,randomly choose one and init
        if isinstance(scenario_parameter, dict):
            scenario = scenario_parameter
        else:  # instance array of dict
            scenario = random.choice(scenario_parameter)

        # if map_name not in (town, "OpenDriveMap"):  TODO
        #     print("The CARLA server uses the wrong map: {}".format(map_name))
        #     print("This scenario requires to use map: {}".format(town))
        #     return False

        self._scenario_map = scenario
        for actor_id, actor in scenario["actors"].items():
            if isinstance(actor["start"], (int, float)):
                self._start_pos[actor_id] = self.pos_coor_map[str(actor["start"])]
            else:
                self._start_pos[actor_id] = actor["start"]

            if isinstance(actor["end"], (int, float)):
                self._end_pos[actor_id] = self.pos_coor_map[str(actor["end"])]
            else:
                self._end_pos[actor_id] = actor["end"]

    def _decode_obs(self, actor_id, obs):
        """Decode actor observation into original image reversing the pre_process() operation.
        Args:
            actor_id (str): Actor identifier
            obs (dict): Properly encoded observation data of an actor

        Returns:
            image (array): Original actor camera view
        """
        if self._actor_configs[actor_id]["send_measurements"]:
            obs = obs[0]
        # Reverse the processing operation
        if self._actor_configs[actor_id]["use_depth_camera"]:
            img = np.tile(obs.swapaxes(0, 1), 3)
        else:
            img = obs.swapaxes(0, 1)
            #img = obs.swapaxes(0, 1) * 128 + 128
        return img

    def _encode_obs(self, actor_id, image, state):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            image (array): original unprocessed image

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        assert self._framestack in [1, 2]
        # Apply preprocessing
        config = self._actor_configs[actor_id]
        image = preprocess_image(image, config)
        # Stack frames
        prev_image = self._prev_image.get(actor_id, None)
        self._prev_image[actor_id] = image
        if prev_image is None:
            prev_image = image
        if self._framestack == 2:
            # image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image])
        del prev_image
        # Structure the observation
        if not self._actor_configs[actor_id]["send_measurements"]:
            return image
        obs = (
            image,
            state,
        )

        return obs

    def step(self, action_dict):
        """Execute one environment step for the specified actors.

        Executes the provided action for the corresponding actors in the
        environment and returns the resulting environment observation, reward,
        done and info (measurements) for each of the actors. The step is
        performed asynchronously i.e. only for the specified actors and not
        necessarily for all actors in the environment.

        Args:
            action_dict (dict): Actions to be executed for each actor. Keys are
                agent_id strings, values are corresponding actions.

        Returns
            obs (dict): Observations for each actor.
            rewards (dict): Reward values for each actor. None for first step
            dones (dict): Done values for each actor. Special key "__all__" is
            set when all actors are done and the env terminates
            info (dict): Info for each actor.

        Raises
            RuntimeError: If `step(...)` is called before calling `reset()`
            ValueError: If `action_dict` is not a dictionary of actions
            ValueError: If `action_dict` contains actions for nonexistent actor
        """

        if not self._carla:
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(self._actors), (
            "No actors exist in the environment. Either"
            " the environment was not properly "
            "initialized using`reset()` or all the "
            "actors have exited. Cannot execute `step()`"
        )

        if not isinstance(action_dict, dict):
            raise ValueError(
                "`step(action_dict)` expected dict of actions. "
                "Got {}".format(type(action_dict))
            )
        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        if not set(action_dict).issubset(set(self._actors)):
            raise ValueError(
                "Cannot execute actions for non-existent actors."
                " Received unexpected actor ids:{}".format(
                    set(action_dict).difference(set(self._actors))
                )
            )

        try:
            #refresh step info
            obs_dict = {}
            reward_dict = {}
            info_dict = {}
            actions = {}

            for actor_id, action in action_dict.items():
                self._auto_controller[actor_id].set_info(
                    {'left_wps': self._state["wps"][actor_id].left_front_wps, 
                     'center_wps': self._state["wps"][actor_id].center_front_wps,
                     'right_wps': self._state["wps"][actor_id].right_front_wps, 
                     'left_rear_wps': self._state["wps"][actor_id].left_rear_wps,
                     'center_rear_wps': self._state["wps"][actor_id].center_rear_wps, 
                     'right_rear_wps': self._state["wps"][actor_id].right_rear_wps,
                     'vehs_info': self._state["vehs"][actor_id]})
                actions.update({actor_id: self._step_before_tick(actor_id, action_dict[actor_id])})

            self._state = {
                "wps":{},   #Dictionary of waypoints info with actor_id as key
                "lights":{}, #Dictionary of traffic lights info with actor_id as key
                "vehs":{},  #Dictionary of other vehicles info with actor_id as key
            }
            self.control_info = {}
            # Asynchronosly (one actor at a time; not all at once in a sync) apply
            # actor actions & perform a server tick after each actor's apply_action
            # if running with sync_server steps
            # NOTE: A distinction is made between "(A)Synchronous Environment" and
            # "(A)Synchronous (carla) server"
            if self._sync_server:
                self._carla.tick(LOG.multi_env_logger)
                if self._render:
                    spectator = self._carla.get_spectator(LOG.multi_env_logger)
                    transform = self._actors[actor_id].get_transform()
                    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
                                                            carla.Rotation(pitch=-90)))
                # `wait_for_tick` is no longer needed, see https://github.com/carla-simulator/carla/pull/1803
                # self.world.wait_for_tick()

            for actor_id, _ in action_dict.items():
                obs, reward, done, truncated, info = self._step_after_tick(actor_id)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                self._done_dict[actor_id] = done
                self._truncated_dict[actor_id] = truncated
                info_dict[actor_id] = info
                if self._speed_state[actor_id] == SpeedState.RUNNING:
                    self._time_steps[actor_id] += 1
                    self._vel_buffer[actor_id].append(self._cur_measurement[actor_id]["velocity"])
                    self._total_steps += 1
                    if self._rl_switch:
                        self._rl_control_steps += 1

            self._done_dict["__all__"] = sum(self._done_dict.values()) >= len(self._actors)
            if len(list(filter(lambda x:  x!=Truncated.FALSE, self._truncated_dict.values()))) > 0:
                self._truncated_dict["__all__"] = Truncated.TRUE

            for actor_id in self._measurements_file_dict:
                if self._actor_configs[actor_id]["log_measurements"] and LOG.log_dir and \
                                (self._truncated_dict["__all__"] == Truncated.TRUE or 
                                self._done_dict["__all__"] == True) and \
                                self._measurements_file_dict[actor_id] is not None:
                    self._measurements_file_dict[actor_id].write("{}]\n")
                    self._measurements_file_dict[actor_id].close()
                    self._measurements_file_dict[actor_id] = None

            # Find if any actor's config has render=True & render only for
            # that actor. NOTE: with async server stepping, enabling rendering
            # affects the step time & therefore MAX_STEPS needs adjustments
            render_required = [
                k for k, v in self._actor_configs.items() if v.get("render", False)]
            if render_required:
                images = {
                    k: self._decode_obs(k, v)
                    for k, v in obs_dict.items()
                    if self._actor_configs[k]["render"]}

                Render.multi_view_render(images, self._camera_poses)
                if self._manual_controller is None:
                    Render.dummy_event_handler()

            if self._verbose:
                print_measurements(LOG.multi_env_logger, self._cur_measurement)
            return obs_dict, reward_dict, self._done_dict, self._truncated_dict, info_dict
        except CarlaError as e:
            LOG.multi_env_logger.exception(e.args)
            self._clear_server_state()
            raise e
        except AttributeError as e:
            LOG.multi_env_logger.exception(e.args)
            if e.args[0].find("'NoneType' object has no attribute") == -1:
                raise e
            else:
                self._clear_server_state()
                raise CarlaError("Carla failed, restart carla!") from e
        except Exception as e:
            LOG.multi_env_logger.exception(f"Error during step, terminating episode early."
                             f"{traceback.format_exc()}")
            self._clear_server_state()
            raise e

    def _step_before_tick(self, actor_id, action):
        """Perform the actual step in the CARLA environment

        Applies control to `actor_id` based on `action`, process measurements,
        compute the rewards and terminal state info (dones).

        Args:
            actor_id(str): Actor identifier
            action:{
                "action_index": 
                "action_param":
            }   Actions to be executed for the actor.
        Returns
            control info (dict)
        """
        if self._discrete_actions:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action[0]) == 2, "Invalid action {}".format(action)
        config = self._actor_configs[actor_id]
        if self._squash_action_logits:
            # forward = 2 * float(sigmoid(action_param[0]) - 0.5)
            # throttle = float(np.clip(forward, 0, 1))
            # brake = float(np.abs(np.clip(forward, -1, 0)))
            # steer = 2 * float(sigmoid(action_param[1]) - 0.5)
            pass
        else:
            steer = action[0][0]
            if action[0][1] >= 0:
                brake = 0
                throttle = np.clip(action[0][1], 0, config["throttle_bound"])
            else:
                throttle = 0
                brake = np.clip(abs(action[0][1]), 0, config["brake_bound"])
        control = ControlInfo(throttle=throttle, brake=brake, steer=steer)

        if config["manual_control"]:
            self._control_clock.tick(60)
            self._manual_control_camera_manager._hud.tick(
                weakref.proxy(self._carla._world),
                self._actors[actor_id],
                self._collisions[actor_id],
                self._control_clock,
            )
            self._manual_controller.parse_events(self, self._control_clock)

            # TODO: consider move this to Render as well
            self._manual_control_camera_manager.render(
                Render.get_screen(), self._manual_control_render_pose
            )
            self._manual_control_camera_manager._hud.render(
                Render.get_screen(), self._manual_control_render_pose
            )
            pygame.display.flip()
        else:
            # TODO: Planner based on waypoints.
            # cur_location = self.actor_list[i].get_location()
            # dst_location = carla.Location(x = self.end_pos[i][0],
            # y = self.end_pos[i][1], z = self.end_pos[i][2])
            # cur_map = self.map
            # next_point_transform = get_transform_from_nearest_way_point(
            # cur_map, cur_location, dst_location)
            # the point with z = 0, and the default z of cars are 40
            # next_point_transform.location.z = 40
            # self.actor_list[i].set_transform(next_point_transform)

            agent_type = config.get("type", "vehicle")
            # TODO: Add proper support for pedestrian actor according to action
            # space of ped actors
            if agent_type == "pedestrian":
                rotation = self._actors[actor_id].get_transform().rotation
                rotation.yaw += steer * 10.0
                x_dir = math.cos(math.radians(rotation.yaw))
                y_dir = math.sin(math.radians(rotation.yaw))

                self._actors[actor_id].apply_control(
                    carla.WalkerControl(
                        speed=3.0 * control.throttle,
                        direction=carla.Vector3D(x_dir, y_dir, 0.0),
                    )
                )

            # TODO: Change this if different vehicle types (Eg.:vehicle_4W,
            #  vehicle_2W, etc) have different control APIs
            elif "vehicle" in agent_type:
                cont = self._speed_switch(actor_id, control)
                if cont is not None:
                    self._actors[actor_id].apply_control(
                        carla.VehicleControl(
                            throttle=float(cont.throttle),
                            steer=float(cont.steer),
                            brake=float(cont.brake),
                            hand_brake=cont.hand_brake,
                            reverse=cont.reverse,
                            manual_gear_shift = cont.manual_gear_shift,
                            gear = int(cont.gear)
                        )
                    )
                    control = cont
        
        return {'steer':control.steer, 'throttle': control.throttle, 'brake': control.brake, 
                'reverse':control.reverse, 'hand_brake':control.hand_brake, 'gear':control.gear}

    def _step_after_tick(self, actor_id):
        """Perform the actual step in the CARLA environment

        process measurements,
        compute the rewards and terminal state info (dones and truncateds).

        Args:
            actor_id(str): Actor identifier

        Returns
            obs (obs_space): Observation for the actor whose id is actor_id.
            reward (float): Reward for actor. None for first step
            done (bool): Done value for actor.
            info (dict): Info for actor.
        """

        # Process observations
        control = self._actors[actor_id].get_control()
        self.control_info[actor_id] = ControlInfo(
            throttle=control.throttle, brake=control.brake, steer=control.steer, gear=control.gear, 
            reverse=control.reverse, hand_brake=control.hand_brake, manual_gear_shift=control.manual_gear_shift)
        config = self._actor_configs[actor_id]
        state = self._read_observation(actor_id)
        py_measurements = self._cur_measurement[actor_id]
        if (self.last_light_state.get(actor_id, None) == carla.TrafficLightState.Red and \
                self._state["lights"][actor_id] is not None and 
                self.last_light_state[actor_id] != self._state["lights"][actor_id].state
                ) or self._state["vehs"][actor_id].center_front_veh is not None:
            #light state change during steps, from red to green or have front vehicle obstacle
            self._vel_buffer[actor_id].clear()

        # Compute truncated
        truncated = self._truncated(actor_id)
        # Compute done
        done = self._done(actor_id, truncated)
        # done = (
        #     self._time_steps[actor_id] > self._scenario_map["max_steps"]
        #     or py_measurements["next_command"] == "REACH_GOAL"
        #     or (
        #         config["early_terminate_on_collision"]
        #         and collided_done(py_measurements)
        #     )
        # )
        py_measurements["done"] = done
        py_measurements["truncated"] = str(truncated)

        # Compute reward
        flag = config["reward_function"]
        if isinstance(self._reward_policy, Reward):
            self._reward_policy.set_state(
                self._actors[actor_id], 
                {
                    "wps": self._state["wps"][actor_id],   
                    "lights": self._state["lights"][actor_id], 
                    "vehs": self._state["vehs"][actor_id],  
                }, 
                weakref.proxy(self._carla._map))
        reward = self._reward_policy.compute_reward(
            actor_id, self._prev_measurement[actor_id], py_measurements, flag
        )
        if self._total_reward[actor_id] is None:
            self._total_reward[actor_id] = reward
        else:
            self._total_reward[actor_id] += reward

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self._total_reward[actor_id]
        py_measurements["reward_info"] = self._reward_policy.info()
        py_measurements["step"]=self._time_steps[actor_id] 

        #update last step info
        lane_center=get_lane_center(weakref.proxy(self._carla._map), self._actors[actor_id].get_location())
        yaw_forward = lane_center.transform.get_forward_vector().make_unit_vector()
        a_3d=self._actors[actor_id].get_acceleration()
        self.last_acc[actor_id],a_t=get_projection(a_3d,yaw_forward)
        self.last_yaw[actor_id] = self._actors[actor_id].get_transform().get_forward_vector()
        self.last_lane[actor_id]=self.current_lane[actor_id]=lane_center.lane_id
        self._prev_measurement[actor_id] = self._cur_measurement[actor_id]
        self._previous_rewards[actor_id] = reward
        if self._state["lights"][actor_id]:
            self.last_light_state[actor_id]=self._state["lights"][actor_id].state
        else:
            self.last_light_state[actor_id]=None

        if config["log_measurements"] and LOG.log_dir:
            # Write out measurements to file
            if self._measurements_file_dict.get(actor_id, None) is None:
                try:
                    self._measurements_file_dict[actor_id] = open(
                        os.path.join(
                            LOG.log_dir,
                            "measurements_{}_{}.json".format(actor_id, self._num_episodes[actor_id]),
                        ),
                        "a",
                    )
                except Exception as e:
                    LOG.multi_env_logger.error(f"File Open Error: {os.path.join(LOG.log_dir, 'measurements_{}_{}.json'.format(actor_id, self._num_episodes[actor_id]))}")
                    raise e
                self._measurements_file_dict[actor_id].write("[\n")
            self._measurements_file_dict[actor_id].write(json.dumps(py_measurements, indent=4))
            self._measurements_file_dict[actor_id].write(",\n")
            if done or truncated!=Truncated.FALSE:
                self._measurements_file_dict[actor_id].write("{}]\n")
                self._measurements_file_dict[actor_id].close()
                self._measurements_file_dict[actor_id] = None
                # if self.config["convert_images_to_video"] and\
                #  (not self.video):
                #    self.images_to_video()
                #    self.video = Trueseg_city_space
        original_image = self._cameras[actor_id].image

        return (
            self._encode_obs(actor_id, original_image, state),
            reward,
            done,
            truncated,
            py_measurements,
        )

    def _read_observation(self, actor_id):
        """Read observation and return measurement.

        Args:
            actor_id (str): Actor identifier

        Returns:
            dict: state dict information containing waypoints, hero vehicle, traffic lights and npc vehicles

        """
        cur = self._actors[actor_id]
        cur_config = self._actor_configs[actor_id]
        planner_enabled = cur_config["enable_planner"]
        if planner_enabled:
            dist = self._path_trackers[actor_id].get_distance_to_end()
            orientation_diff = self._path_trackers[
                actor_id
            ].get_orientation_difference_to_end_in_radians()
            commands = self.planner.plan_route(
                (cur.get_location().x, cur.get_location().y),
                (self._end_pos[actor_id][0], self._end_pos[actor_id][1]),
            )
            if len(commands) > 0:
                next_command = ROAD_OPTION_TO_COMMANDS_MAPPING.get(
                    commands[0], "LANE_FOLLOW"
                )
            elif (
                dist <= DISTANCE_TO_GOAL_THRESHOLD
                and orientation_diff <= ORIENTATION_TO_GOAL_THRESHOLD
            ):
                next_command = "REACH_GOAL"
            else:
                next_command = "LANE_FOLLOW"

            # DEBUG
            # self.path_trackers[actor_id].draw()
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self._collisions[actor_id].collision_vehicles
        collision_pedestrians = self._collisions[actor_id].collision_pedestrians
        collision_other = self._collisions[actor_id].collision_other
        intersection_otherlane = self._lane_invasions[actor_id].offlane
        intersection_offroad = self._lane_invasions[actor_id].offroad

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0
        elif planner_enabled:
            distance_to_goal = self._path_trackers[actor_id].get_distance_to_end()
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm(
                [
                    self._actors[actor_id].get_location().x
                    - self._end_pos[actor_id][0],
                    self._actors[actor_id].get_location().y
                    - self._end_pos[actor_id][1],
                ]
            )
        )

        # get surrounding information
        state, measurement, state_np = self._state_getter.get_state(actor_id)
        self._state["wps"].update({actor_id: state["wps"]})
        self._state["lights"].update({actor_id: state["lights"]})
        self._state["vehs"].update({actor_id: state["vehs"]})

        self._cur_measurement.update({actor_id: {
            # "x": self._actors[actor_id].get_location().x,
            # "y": self._actors[actor_id].get_location().y,
            # "pitch": self._actors[actor_id].get_transform().rotation.pitch,
            # "yaw": self._actors[actor_id].get_transform().rotation.yaw,
            # "roll": self._actors[actor_id].get_transform().rotation.roll,
            "distance_to_goal": distance_to_goal,
            # "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            # "weather": self._weather,
            # "map": self._server_map,
            # "start_coord": self._start_coord[actor_id],
            # "end_coord": self._end_coord[actor_id],
            # "current_scenario": self._scenario_map,
            # "x_res": self._x_res,
            # "y_res": self._y_res,
            # "max_steps": self._scenario_map["max_steps"],
            # "next_command": next_command,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "control_info": self.control_info.get(actor_id, ControlInfo()).toDict(),
            "previous_reward": self._previous_rewards.get(actor_id, None),
            "speed_state": str(self._speed_state[actor_id]),
            "rl_switch": self._rl_switch,
            "step": self._time_steps[actor_id],
            "episode": self._num_episodes[actor_id],
            "last_acc": self.last_acc[actor_id],
            "last_yaw": {
                "x": self.last_yaw[actor_id].x,
                "y": self.last_yaw[actor_id].y,
                "z": self.last_yaw[actor_id].z,
            },
            "last_lane": self.last_lane[actor_id],
            "current_lane": self.current_lane[actor_id],
            "last_light_state": self.last_light_state[actor_id],
            "velocity": measurement["velocity"],
            "current_acc": measurement["current_acc"],
            "rear_id": measurement["rear_id"],
            "rear_v": measurement["rear_v"],
            "rear_a": measurement["rear_a"],
            "change_lane": measurement["change_lane"]
        }})

        return state_np
        
    def _speed_switch(self, actor_id, cont):
        """cont: the control command of RL agent"""
        ego_speed = get_speed(self._actors[actor_id])
        if self._speed_state[actor_id] == SpeedState.START:
            hero_autopilot(self._actors[actor_id], weakref.proxy(self._carla._traffic_manager), 
                           self._actor_configs[actor_id], self._env_config, True)
            control = None
            # control = self.controller.run_step({'waypoints':self.next_wps,'vehicle_front':self.vehicle_front})
            if ego_speed >= self._actor_configs[actor_id]["speed_threshold"]:
                self._speed_state[actor_id] = SpeedState.RUNNING
                if self._actor_configs[actor_id]["auto_control"]:
                    hero_autopilot(self._actors[actor_id], weakref.proxy(self._carla._traffic_manager), 
                                self._actor_configs[actor_id], self._env_config, True)
                    control = None
                else:
                    if self._rl_switch:
                        hero_autopilot(self._actors[actor_id], weakref.proxy(self._carla._traffic_manager), 
                                self._actor_configs[actor_id], self._env_config, False)
                        # RL in control       
                        control = cont
                    else:
                        # traffic manager in control
                        control = None         
        elif self._speed_state[actor_id] == SpeedState.RUNNING:
            if self._state["wps"][actor_id].center_front_wps[2].road_id == self._scenario_map["dest_road_id"]:          
                #Vehicle reaches destination, stop vehicle
                self._speed_state[actor_id] = SpeedState.STOP
                self._collisions[actor_id].sensor.stop()
                self._lane_invasions[actor_id].sensor.stop()
                hero_autopilot(self._actors[actor_id], weakref.proxy(self._carla._traffic_manager), 
                               self._actor_configs[actor_id], self._env_config, True)
                control = None
            elif not self._actor_configs[actor_id]["auto_control"]:
                if self._rl_switch:
                    # under Rl control
                    control = cont
                else:
                    # traffic manager in control
                    control = None
        elif self._speed_state[actor_id] == SpeedState.STOP:
            #Hero vehicle reaches destination, properly stop hero vehicle
            self._carla._traffic_manager.vehicle_percentage_speed_difference(self._actors[actor_id], 90)
            control = None
        else:
            LOG.multi_env_logger.error('CODE LOGIC ERROR')

        return control

    def _truncated(self, actor_id):
        """Decide whether to terminate current episode or not"""
        m = self._cur_measurement[actor_id]
        lane_center=get_lane_center(weakref.proxy(self._carla._map), self._actors[actor_id].get_location())
        yaw_diff = math.degrees(get_yaw_diff(lane_center.transform.get_forward_vector(),
                        self._actors[actor_id].get_transform().get_forward_vector()))
        
        if self._speed_state[actor_id] == SpeedState.STOP:
            LOG.multi_env_logger.info(actor_id + " vehicle reach destination, stop truncation")
            return Truncated.FALSE
        if (m["collision_vehicles"] > 0 or m["collision_pedestrians"]>0 or m["collision_other"] > 0) \
                and self._actor_configs[actor_id].get("early_terminate_on_collision", True):
            # Here we judge speed state because there might be collision event when spawning vehicles
                    # Ignore case in which another actor collides with ego actor from the back
            history, tags, ids = self._collisions[actor_id].get_collision_history()
            collision = True
            for id in ids:
                actor = self._carla._world.get_actor(id)
                actor_lane = get_lane_center(weakref.proxy(self._carla._map), actor.get_location())
                if self._state["vehs"][actor_id].center_rear_veh and \
                        self._state["vehs"][actor_id].center_rear_veh.id == id:
                    # Ignore the case in which other actor collide with hero vehicle from the back
                    collision = False
                    self._collisions[actor_id]._reset()
            
            if collision:
                LOG.multi_env_logger.warn(actor_id + ' collison happend')
                return Truncated.COLLISION
        if not test_waypoint(lane_center,False):
            LOG.multi_env_logger.warn(actor_id + ' vehicle drive out of road')
            return Truncated.OUT_OF_ROAD
        if self._speed_state[actor_id] not in [SpeedState.START, SpeedState.STOP] \
                    and not self._state["vehs"][actor_id].center_front_veh:
            if not self._state["lights"][actor_id] or self._state["lights"][actor_id].state!=carla.TrafficLightState.Red:
                if len(self._vel_buffer[actor_id]) == self._vel_buffer[actor_id].maxlen:
                    avg_vel=0
                    for vel in self._vel_buffer[actor_id]:
                        avg_vel += vel / self._vel_buffer[actor_id].maxlen
                    if avg_vel*3.6 < self._actor_configs[actor_id]["speed_min"]:
                        LOG.multi_env_logger.warn(actor_id + ' vehicle speed too low')
                        return Truncated.SPEED_LOW
            
        # if self.lane_invasion_sensor.get_invasion_count()!=0:
        #     LOG.multi_env_logger.warn('lane invasion occur')
        #     return True
        if abs(yaw_diff)>90:
            LOG.multi_env_logger.warn(actor_id + ' moving in opposite direction')
            return Truncated.OPPOSITE_DIRECTION
        if self._state["lights"][actor_id] and self._state["lights"][actor_id].state!=carla.TrafficLightState.Green:
            self._carla._world.debug.draw_point(self._state["lights"][actor_id].get_location(),size=0.3,life_time=0)
            wps=self._state["lights"][actor_id].get_stop_waypoints()
            for wp in wps:
                self._carla._world.debug.draw_point(wp.transform.location,size=0.1,life_time=0)
                if is_within_distance_ahead(self._actors[actor_id].get_location(),
                        wp.transform.location, wp.transform, self._env_config["min_distance"]):
                    LOG.multi_env_logger.warn(actor_id + ' break traffic light rule')
                    return Truncated.TRAFFIC_LIGHT_BREAK

        return Truncated.FALSE

    def _done(self, actor_id, truncated):
        if truncated!=Truncated.FALSE:
            return False
        if self._speed_state[actor_id] == SpeedState.STOP:
            LOG.multi_env_logger.info('vehicle reach destination, stop '+str(actor_id))                              
            return True
        if not self._rl_switch:
            if self._time_steps[actor_id] > self._scenario_config["max_steps"]:
                # Let the traffic manager only execute 5000 steps. or it can fill the replay buffer
                LOG.multi_env_logger.info(f"{self._scenario_config['max_steps']} "
                             f"steps passed under traffic manager control, stop "+actor_id)
                return True

        return False

    def close(self):
        """Clean-up the world, clear server state & close the Env"""
        self._clean_world()
        self._clear_server_state()

    def log(self, msg:str, level:str):
        level = level.upper
        if level is "DEBUG":
            LOG.rl_trainer_logger.debug(msg)
        elif level is "INFO":
            LOG.rl_trainer_logger.info(msg)
        elif level is "WARNING" or "WARN":
            LOG.rl_trainer_logger.warn(msg)
        elif level is "EXCEPTION":
            LOG.rl_trainer_logger.exception(msg)
        elif level is "ERROR":
            LOG.rl_trainer_logger.error(msg)
        elif level is "CRITICAL":
            LOG.rl_trainer_logger.critical(msg)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument("--scenario", default="3", help="print debug information")
    # TODO: Fix the default path to the config.json;Should work after packaging
    argparser.add_argument(
        "--config",
        default="src/macad_gym/carla/config.json",
        help="print debug information",
    )

    argparser.add_argument("--map", default="Town01", help="print debug information")

    args = argparser.parse_args()

    multi_env_config = json.load(open(args.config))
    env = MultiCarlaEnv(multi_env_config)

    for _ in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = multi_env_config["env"]
        actor_configs = multi_env_config["actors"]
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
            action_dict = get_next_actions(info, env._discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            LOG.multi_env_logger.info(
                ":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                    i, reward, total_reward_dict, done
                )
            )

        LOG.multi_env_logger.info("{} fps".format(i / (time.time() - start)))
