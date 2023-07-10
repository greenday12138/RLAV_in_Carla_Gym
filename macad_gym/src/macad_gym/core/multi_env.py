"""
multi_env.py: Multi-actor environment interface for CARLA-Gym
Should support two modes of operation. See CARLA-Gym developer guide for
more information
__author__: @Praveen-Palanisamy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import atexit
import shutil
from datetime import datetime
import logging
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import socket
import math

import numpy as np  # linalg.norm is used
import GPUtil
from gym.spaces import Box, Discrete, Tuple, Dict
import pygame
import carla

from macad_gym.core.controllers.traffic import apply_traffic, hero_autopilot
from macad_gym.multi_actor_env import MultiActorEnv
from macad_gym import LOG_DIR
from macad_gym.core.sensors.utils import preprocess_image
from macad_gym.core.maps.nodeid_coord_map import MAP_TO_COORDS_MAPPING
from macad_gym.core.utils.misc import (remove_unnecessary_objects, sigmoid, get_lane_center, 
    get_yaw_diff, test_waypoint, is_within_distance_ahead, get_projection, draw_waypoints,
    get_speed)
from macad_gym.core.utils.wrapper import (COMMANDS_ENUM, COMMAND_ORDINAL, ROAD_OPTION_TO_COMMANDS_MAPPING, 
    DISTANCE_TO_GOAL_THRESHOLD, ORIENTATION_TO_GOAL_THRESHOLD, RETRIES_ON_ERROR, GROUND_Z, DISCRETE_ACTIONS,
    WEATHERS, process_lane_wp, process_veh, get_next_actions, DEFAULT_MULTIENV_CONFIG, print_measurements,
    Truncated, Action, SpeedState, ControlInfo)

# from macad_gym.core.sensors.utils import get_transform_from_nearest_way_point
from macad_gym.core.reward import Reward, PDQNReward
from macad_gym.core.sensors.hud import HUD
from macad_gym.viz.render import Render
from macad_gym.core.scenarios import Scenarios

# The following imports require carla to be imported already.
from macad_gym.core.sensors.camera_manager import CameraManager, CAMERA_TYPES
from macad_gym.core.sensors.derived_sensors import LaneInvasionSensor
from macad_gym.core.sensors.derived_sensors import CollisionSensor
from macad_gym.core.controllers.keyboard_control import KeyboardControl
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner_dao import (  # noqa: E501
    GlobalRoutePlannerDAO)
from macad_gym.core.controllers.route_planner import RoutePlanner
from macad_gym.core.controllers.local_planner import LocalPlanner

# The following imports depend on these paths being in sys path
# TODO: Fix this. This probably won't work after packaging/distribution
sys.path.append("src/macad_gym/carla/PythonAPI")
from macad_gym.core.maps.nav_utils import PathTracker  # noqa: E402
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner import (  # noqa: E402, E501
    GlobalRoutePlanner)

logger = logging.getLogger(__name__)
# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/Git/RLAV_in_Carla_Gym/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/ProgramFiles/Carla/CarlaUE4.sh")
)

# Check if is using on Windows
IS_WINDOWS_PLATFORM = "win" in sys.platform

assert os.path.exists(SERVER_BINARY), (
    "Make sure CARLA_SERVER environment"
    " variable is set & is pointing to the"
    " CARLA server startup script (Carla"
    "UE4.sh). Refer to the README file/docs."
)

live_carla_processes = set()


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        if IS_WINDOWS_PLATFORM:
            # for Windows
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(pgid)])
        else:
            # for Linux
            os.killpg(pgid, signal.SIGKILL)

    live_carla_processes.clear()


def termination_cleanup(*_):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(cleanup)

MultiAgentEnvBases = [MultiActorEnv]
try:
    from ray.rllib.env import MultiAgentEnv

    MultiAgentEnvBases.append(MultiAgentEnv)
except ImportError:
    logger.warning("\n Disabling RLlib support.", exc_info=True)



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
        self._reward_policy = Reward()
        configs["scenarios"] = Scenarios.resolve_scenarios_parameter(
            configs["scenarios"]
        )

        self._scenario_config = configs["scenarios"]
        self._env_config = configs["env"]
        self._actor_configs = configs["actors"]
        self._rl_configs = configs["rl_parameters"]

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
                            self._image_space,  # image
                            Discrete(len(COMMANDS_ENUM)),  # next_command
                            Box(
                                -128.0, 128.0, shape=(2,)
                            ),  # forward_speed, dist to goal
                        ]
                    )
                    for actor_id in self._actor_configs.keys()
                }
            )
        else:
            self.observation_space = Dict(
                {actor_id: self._image_space for actor_id in self._actor_configs.keys()}
            )

        # Set appropriate node-id to coordinate mappings for Town01 or Town02.
        self.pos_coor_map = MAP_TO_COORDS_MAPPING[self._server_map]
        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._server_port = None
        self._server_process = None
        self._client = None
        self._num_steps = {}
        self._num_episodes = {}
        self._total_reward = {}
        self._prev_measurement = {}
        self._prev_image = None
        self._measurements_file_dict = {}
        self._weather = None
        self._start_pos = {}  # Start pose for each actor
        self._end_pos = {}  # End pose for each actor
        self._start_coord = {}
        self._end_coord = {}
        self._last_obs = None
        self._image = None
        self._surface = None
        self._video = False
        self._obs_dict = {}
        self._done_dict = {}
        self._truncated_dict = {}
        self._previous_actions = {}
        self._previous_rewards = {}
        self._last_reward = {}
        self._npc_vehicles = []  # List of NPC vehicles
        self._npc_vehicles_spawn_points = []
        self._npc_pedestrians = []  # List of NPC pedestrians
        self._agents = {}  # Dictionary of macad_agents with agent_id as key
        self._actors = {}  # Dictionary of actors with actor_id as key
        self._speed_state = {}  # Dictionary of actors' running state with actor_id as key
        self._local_planner = {}
        self._state = {
            "wps":{},   #Dictionary of waypoints info with actor_id as keyw
            "lights":{}, #Dictionary of traffic lights info with actor_id as key
            "vehs":{},  #Dictionary of other vehicles info with actor_id as key
            "meas":{},   #Dictionary of actor's py_measurements with actor_id as key
        }
        self._cameras = {}  # Dictionary of sensors with actor_id as key
        self._path_trackers = {}  # Dictionary of sensors with actor_id as key
        self._collisions = {}  # Dictionary of sensors with actor_id as key
        self._lane_invasions = {}  # Dictionary of sensors with actor_id as key
        self._scenario_map = {}  # Dictionary with current scenario map config
        # rl_switch: True--currently RL in control, False--currently PID in control
        self._rl_switch = {}
        self.switch_count=0
        self.SWITCH_THRESHOLD = self._rl_configs["switch_threshold"]
        self.pre_train_steps = self._rl_configs["pre_train_steps"]

    @staticmethod
    def _get_tcp_port(port=0):
        """
        Get a free tcp port number
        :param port: (default 0) port number. When `0` it will be assigned a free port dynamically
        :return: a port number requested if free otherwise an unhandled exception would be thrown
        """
        s = socket.socket()
        s.bind(("", port))
        server_port = s.getsockname()[1]
        s.close()
        return server_port

    def _init_server(self):
        """Initialize carla server and client

        Returns:
            N/A
        """
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        # First find a port that is free and then use it in order to avoid
        # crashes due to:"...bind:Address already in use"
        self._server_port = self._get_tcp_port()

        multigpu_success = False
        gpus = GPUtil.getGPUs()
        log_file = os.path.join(LOG_DIR, "server_" + str(self._server_port) + ".log")
        logger.info(
            f"1. Port: {self._server_port}\n"
            f"2. Map: {self._server_map}\n"
            f"3. Binary: {SERVER_BINARY}"
        )

        if not self._render and (gpus is not None and len(gpus)) > 0:
            try:
                min_index = random.randint(0, len(gpus) - 1)
                for i, gpu in enumerate(gpus):
                    if gpu.load < gpus[min_index].load:
                        min_index = i
                # Check if vglrun is setup to launch sim on multipl GPUs
                if shutil.which("vglrun") is not None:
                    self._server_process = subprocess.Popen(
                        (
                            "DISPLAY=:8 vglrun -d :7.{} {} -benchmark -fps={}"
                            " -carla-server -world-port={}"
                            " -carla-streaming-port=0".format(
                                min_index,
                                SERVER_BINARY,
                                1/self._fixed_delta_seconds,
                                self._server_port,
                            )
                        ),
                        shell=True,
                        # for Linux
                        preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                        # for Windows (not necessary)
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                        if IS_WINDOWS_PLATFORM
                        else 0,
                        stdout=open(log_file, "w"),
                    )

                # Else, run in headless mode
                else:
                    # Since carla 0.9.12+ use -RenderOffScreen to start headlessly
                    # https://carla.readthedocs.io/en/latest/adv_rendering_options/
                    self._server_process = subprocess.Popen(
                        (  # 'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE={} DISPLAY='
                            '"{}" -RenderOffScreen -benchmark -fps={} -carla-server'
                            " -world-port={} -carla-streaming-port=0".format(
                                SERVER_BINARY,
                                1/self._fixed_delta_seconds,
                                self._server_port,
                            )
                        ),
                        shell=True,
                        # for Linux
                        preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                        # for Windows (not necessary)
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                        if IS_WINDOWS_PLATFORM
                        else 0,
                        stdout=open(log_file, "w"),
                    )
            # TODO: Make the try-except style handling work with Popen
            # exceptions after launching the server procs are not caught
            except Exception as e:
                print(e)
            # Temporary soln to check if CARLA server proc started and wrote
            # something to stdout which is the usual case during startup
            if os.path.isfile(log_file):
                multigpu_success = True
            else:
                multigpu_success = False

            if multigpu_success:
                print("Running sim servers in headless/multi-GPU mode")

        # Rendering mode and also a fallback if headless/multi-GPU doesn't work
        if multigpu_success is False:
            try:
                print("Using single gpu to initialize carla server")

                self._server_process = subprocess.Popen(
                    [
                        SERVER_BINARY,
                        "-windowed",
                        #"-prefernvidia",
                        "-quality-level=Low",
                        "-ResX=",
                        str(self._env_config["render_x_res"]),
                        "-ResY=",
                        str(self._env_config["render_y_res"]),
                        "-benchmark",
                        "-fps={}".format(1/self._fixed_delta_seconds),
                        "-carla-server",
                        "-carla-rpc-port={}".format(self._server_port),
                        "-carla-streaming-port=0",
                    ],
                    # for Linux
                    preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                    # for Windows (not necessary)
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    if IS_WINDOWS_PLATFORM
                    else 0,
                    stdout=open(log_file, "w"),
                    #bufsize=131072
                )
                print("Running simulation in single-GPU mode")
            except Exception as e:
                logger.debug(e)
                print("FATAL ERROR while launching server:", sys.exc_info()[0])

        if IS_WINDOWS_PLATFORM:
            live_carla_processes.add(self._server_process.pid)
        else:
            live_carla_processes.add(os.getpgid(self._server_process.pid))

        # Start client
        self._client = None
        while self._client is None:
            try:
                self._client = carla.Client("localhost", self._server_port)
                # The socket establishment could takes some time
                time.sleep(2)
                self._client.set_timeout(2.0)
                print(
                    "Client successfully connected to server, Carla-Server version: ",
                    self._client.get_server_version(),
                )
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                self._client = None

        self._client.set_timeout(60.0)
        # load map using client api since 0.9.6+
        self._client.load_world(self._server_map)
        self.world = self._client.get_world()
        remove_unnecessary_objects(self.world)
        self.map = self.world.get_map()
        world_settings = self.world.get_settings()
        # Actors will become dormant 2km away from here vehicle
        if self._env_config["hybrid"]:
            world_settings.actor_active_distance = 2000 
        world_settings.synchronous_mode = self._sync_server
        world_settings.no_rendering_mode = not self._render
        if self._sync_server:
            # Synchronous mode
            # try:
            # Available with CARLA version>=0.9.6
            # Set fixed_delta_seconds to have reliable physics between sim steps
            world_settings.fixed_delta_seconds = self._fixed_delta_seconds
        self.world.apply_settings(world_settings)
        # Sign on traffic manager
        self._traffic_manager = self._client.get_trafficmanager()
        self._traffic_manager.set_synchronous_mode(self._sync_server)
        # Set the spectator/server view if rendering is enabled
        if self._render and self._env_config.get("spectator_loc"):
            spectator = self.world.get_spectator()
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
                RoutePlanner(self.map, sampling_resolution=4.0).get_spawn_points()

    def _clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A
        """
        for colli in self._collisions.values():
            if colli.sensor.is_alive:
                colli.sensor.destroy()
        for lane in self._lane_invasions.values():
            if lane.sensor.is_alive:
                lane.sensor.destroy()
        for actor in self._actors.values():
            if actor.is_alive:
                actor.destroy()
        for npc in self._npc_vehicles:
            npc.destroy()
        for npc in zip(*self._npc_pedestrians):
            npc[1].stop()  # stop controller
            npc[0].destroy()  # kill entity
        # Note: the destroy process for cameras is handled in camera_manager.py

        self._cameras = {}
        self._actors = {}
        self._npc_vehicles = []
        self._npc_pedestrians = []
        self._path_trackers = {}
        self._collisions = {}
        self._lane_invasions = {}

        print("Cleaned-up the world...")

    def _clear_server_state(self):
        """Clear server process"""
        print("Clearing Carla server state")
        try:
            if self._client:
                self._client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))
        if self._server_process:
            if IS_WINDOWS_PLATFORM:
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(self._server_process.pid)]
                )
                live_carla_processes.remove(self._server_process.pid)
            else:
                pgid = os.getpgid(self._server_process.pid)
                os.killpg(pgid, signal.SIGKILL)
                live_carla_processes.remove(pgid)

            self._server_port = None
            self._server_process = None

    def reset(self, seed=None, options=None):
        """Reset the carla world, call _init_server()

        Returns:
            N/A
        """
        # World reset and new scenario selection if multiple are available
        self._load_scenario(self._scenario_config)

        for retry in range(RETRIES_ON_ERROR):
            try:
                if not self._server_process:
                    self._init_server()
                    self._reset(clean_world=False)
                else:
                    self._reset()
                break
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                print("reset(): Retry #: {}/{}".format(retry + 1, RETRIES_ON_ERROR))
                self._clear_server_state()
                raise e

        # Set appropriate initial values for all actors
        for actor_id, actor_config in self._actor_configs.items():
            if self._done_dict.get(actor_id, True):
                self._last_reward[actor_id] = None
                self._total_reward[actor_id] = None
                self._num_steps[actor_id] = 0

                cam = self._cameras[actor_id]
                # Wait for the sensor (camera) actor to start streaming
                # Shouldn't take too long
                while cam.callback_count == 0:
                    if self._sync_server:
                        self.world.tick()
                        # `wait_for_tick` is no longer needed, see https://github.com/carla-simulator/carla/pull/1803
                        # self.world.wait_for_tick()
                if cam.image is None:
                    print("callback_count:", actor_id, ":", cam.callback_count)
                state = self._read_observation(actor_id)
                obs = self._encode_obs(actor_id, cam.image, state)
                self._obs_dict[actor_id] = obs
                # Actor correctly reset
                self._done_dict[actor_id] = False
                self._truncated_dict[actor_id] = Truncated.FALSE

        #update prev step info
        for actor_id,_ in self._actor_configs.items():
            self._prev_measurement[actor_id]=self._state["meas"][actor_id]
            #self._prev_measurement.update({actor_id:self._state["meas"][actor_id]})

        return self._obs_dict, _

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
            print("Unsupported actor type:{}. Using vehicle_4W as the type")
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
                self.map.get_waypoint(loc, project_to_road=True).transform.rotation)
            #: If yaw is provided in addition to (X, Y, Z), set yaw
            if len(self._start_pos[actor_id]) > 3:
                rot.yaw = self._start_pos[actor_id][3]
            transform = carla.Transform(loc, rot)
            self._actor_configs[actor_id]["start_transform"] = transform
            tls = traffic_lights.get_tls(self.world, transform, sort=True)
            return tls[0][0]  #: Return the key (carla.TrafficLight object) of
            #: closest match

        if actor_type == "pedestrian":
            blueprints = self.world.get_blueprint_library().filter(
                "walker.pedestrian.*"
            )

        elif actor_type == "vehicle_4W":
            blueprints = self.world.get_blueprint_library().filter("vehicle")
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
            blueprints = self.world.get_blueprint_library().filter("vehicle")
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
            z=self._start_pos[actor_id][2]+1,
        )
        rot = (
            self.map.get_waypoint(loc, project_to_road=True).transform.rotation)
        #: If yaw is provided in addition to (X, Y, Z), set yaw
        if len(self._start_pos[actor_id]) > 3:
            rot.yaw = self._start_pos[actor_id][3]
        transform = carla.Transform(loc, rot)
        draw_waypoints(self.world, [self.map.get_waypoint(transform.location)], life_time=0)
        self._actor_configs[actor_id]["start_transform"] = transform
        vehicle = None
        for retry in range(RETRIES_ON_ERROR):
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if self._sync_server:
                self.world.tick()
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
            print("spawn_actor: Retry#:{}/{}".format(retry + 1, RETRIES_ON_ERROR))
        if vehicle is None:
            # Request a spawn one last time possibly raising the error
            vehicle = self.world.spawn_actor(blueprint, transform)
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

        weather_num = 0
        if "weather_distribution" in self._scenario_map:
            weather_num = random.choice(self._scenario_map["weather_distribution"])
            if weather_num not in WEATHERS:
                weather_num = 0

        self.world.set_weather(WEATHERS[weather_num])

        self._weather = [
            self.world.get_weather().cloudiness,
            self.world.get_weather().precipitation,
            self.world.get_weather().precipitation_deposits,
            self.world.get_weather().wind_intensity,
        ]

        for actor_id, actor_config in self._actor_configs.items():
            if actor_id in self._num_episodes.keys():
                self._num_episodes[actor_id]+=1
            else:
                self._num_episodes[actor_id]=0

            if self._done_dict.get("__all__", True) or \
                    self._truncated_dict.get("__all__", Truncated.FALSE)!=Truncated.FALSE:
                self._measurements_file_dict[actor_id] = None
                actor_config = self._actor_configs[actor_id]

                # Try to spawn actor (soft reset) or fail and reinitialize the server before get back here
                try:
                    self._actors[actor_id] = self._spawn_new_actor(actor_id)
                    self._local_planner[actor_id] = LocalPlanner(
                        self._actors[actor_id], {
                            'sampling_resolution': self._env_config["sampling_resolution"],
                            'buffer_size': self._env_config["buffer_size"],
                            'vehicle_proximity': self._env_config["vehicle_proximity"],
                            'traffic_light_proximity':self._env_config["traffic_light_proximity"]})
                    self._speed_state[actor_id] = SpeedState.START
                    self._rl_switch[actor_id] = False
                except RuntimeError as spawn_err:
                    del self._done_dict[actor_id]
                    del self._truncated_dict[actor_id]
                    # Chain the exception & re-raise to be handled by the caller `self.reset()`
                    raise spawn_err from RuntimeError(
                        "Unable to spawn actor:{}".format(actor_id))

                if self._env_config["enable_planner"]:
                    self._path_trackers[actor_id] = PathTracker(
                        self.world,
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
                    CAMERA_TYPES[camera_type].value - 1, int(camera_pos), notify=False
                )
                assert camera_manager.sensor.is_listening
                self._cameras.update({actor_id: camera_manager})

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

                print(
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
        print("New episode initialized with actors:{}".format(self._actors.keys()))

        self._npc_vehicles, self._npc_pedestrians = apply_traffic(
            self.world,
            self._traffic_manager,
            self._env_config,
            self._scenario_map.get("num_vehicles", 0),
            self._scenario_map.get("num_pedestrians", 0),
            safe=False,
            route_points=self._npc_vehicles_spawn_points
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
            img = obs.swapaxes(0, 1) * 128 + 128
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
        py_measurements = self._state["meas"][actor_id]
        # Apply preprocessing
        config = self._actor_configs[actor_id]
        image = preprocess_image(image, config)
        # Stack frames
        prev_image = self._prev_image
        self._prev_image = image
        if prev_image is None:
            prev_image = image
        if self._framestack == 2:
            # image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image])
        # Structure the observation
        if not self._actor_configs[actor_id]["send_measurements"]:
            return image
        obs = (
            image,
            state,
            COMMAND_ORDINAL[py_measurements["next_command"]],
            [py_measurements["velocity"], py_measurements["distance_to_goal"]],
        )

        self._last_obs = obs
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

        if (not self._server_process) or (not self._client):
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
            obs_dict = {}
            reward_dict = {}
            info_dict = {}
            actions = {}

            for actor_id, action in action_dict.items():
                actions.update({actor_id: self._step_before_tick(actor_id, action)})

            # Asynchronosly (one actor at a time; not all at once in a sync) apply
            # actor actions & perform a server tick after each actor's apply_action
            # if running with sync_server steps
            # NOTE: A distinction is made between "(A)Synchronous Environment" and
            # "(A)Synchronous (carla) server"
            if self._sync_server:
                self.world.tick()
                if self._render:
                    spectator = self.world.get_spectator()
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

            self._done_dict["__all__"] = sum(self._done_dict.values()) >= len(self._actors)
            if len(list(filter(lambda x:  x!=Truncated.FALSE, self._truncated_dict.values()))) > 0:
                self._truncated_dict["__all__"] = Truncated.TRUE
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

            print_measurements(self._state["meas"])
            return obs_dict, reward_dict, self._done_dict["__all__"]!=False, \
                    self._truncated_dict["__all__"]!=Truncated.FALSE, info_dict
        except Exception:
            print(
                "Error during step, terminating episode early.", traceback.format_exc()
            )
            self._clear_server_state()

    def _step_before_tick(self, actor_id, action):
        """Perform the actual step in the CARLA environment

        Applies control to `actor_id` based on `action`, process measurements,
        compute the rewards and terminal state info (dones).

        Args:
            actor_id(str): Actor identifier
            action: Actions to be executed for the actor.

        Returns
            ation (dict)
        """
        if self._discrete_actions:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action) == 2, "Invalid action {}".format(action)
        if self._squash_action_logits:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 0.6))
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))
        control = ControlInfo(throttle=throttle, brake=brake, steer=steer)
        # if self._verbose:
        #     print(
        #         "steer", steer, "throttle", throttle, "brake", brake, "reverse", reverse
        #     )

        config = self._actor_configs[actor_id]
        if config["manual_control"]:
            self._control_clock.tick(60)
            self._manual_control_camera_manager._hud.tick(
                self.world,
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
                control = self._speed_switch(actor_id, control)
                self._actors[actor_id].apply_control(
                    carla.VehicleControl(
                        throttle=control.throttle,
                        steer=control.steer,
                        brake=control.brake,
                        hand_brake=control.hand_brake,
                        reverse=control.reverse,
                    )
                )
        
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
        config = self._actor_configs[actor_id]
        state = self._read_observation(actor_id)
        py_measurements = self._state["meas"][actor_id]
        control = self._actors[actor_id].get_control()
        # if self._verbose:
        #     print("Next command", py_measurements["next_command"])
        # Store previous action
        self._previous_actions[actor_id] = ControlInfo(
            throttle=control.throttle, brake=control.brake, steer=control.steer, gear=control.gear, 
            reverse=control.reverse, hand_brake=control.hand_brake, manual_gear_shift=control.manual_gear_shift)
        py_measurements["control"] = control

        # Compute truncated
        truncated = self._truncated(actor_id)
        # Compute done
        done = self._done(actor_id, truncated)
        # done = (
        #     self._num_steps[actor_id] > self._scenario_map["max_steps"]
        #     or py_measurements["next_command"] == "REACH_GOAL"
        #     or (
        #         config["early_terminate_on_collision"]
        #         and collided_done(py_measurements)
        #     )
        # )
        py_measurements["done"] = done
        py_measurements["truncated"] = truncated

        # Compute reward
        flag = config["reward_function"]
        reward = self._reward_policy.compute_reward(
            self._prev_measurement[actor_id], py_measurements, flag
        )

        self._previous_rewards[actor_id] = reward
        if self._total_reward[actor_id] is None:
            self._total_reward[actor_id] = reward
        else:
            self._total_reward[actor_id] += reward

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self._total_reward[actor_id]

        # End iteration updating parameters and logging
        self._prev_measurement[actor_id] = py_measurements
        self._num_steps[actor_id] += 1

        if config["log_measurements"] and CARLA_OUT_PATH:
            # Write out measurements to file
            if not self._measurements_file_dict[actor_id]:
                self._measurements_file_dict[actor_id] = open(
                    os.path.join(
                        CARLA_OUT_PATH,
                        "measurements_{}.json".format(self._num_episodes[actor_id]),
                    ),
                    "w",
                )
            self._measurements_file_dict[actor_id].write(json.dumps(py_measurements))
            self._measurements_file_dict[actor_id].write("\n")
            if done:
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

        self._state["meas"].update({actor_id: {
            "episode": self._num_episodes[actor_id],
            "step": self._num_steps[actor_id],
            "x": self._actors[actor_id].get_location().x,
            "y": self._actors[actor_id].get_location().y,
            "pitch": self._actors[actor_id].get_transform().rotation.pitch,
            "yaw": self._actors[actor_id].get_transform().rotation.yaw,
            "roll": self._actors[actor_id].get_transform().rotation.roll,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            "weather": self._weather,
            "map": self._server_map,
            "start_coord": self._start_coord[actor_id],
            "end_coord": self._end_coord[actor_id],
            "current_scenario": self._scenario_map,
            "x_res": self._x_res,
            "y_res": self._y_res,
            "max_steps": self._scenario_map["max_steps"],
            "next_command": next_command,
            "previous_action": self._previous_actions.get(actor_id, None),
            "previous_reward": self._previous_rewards.get(actor_id, None),
            "speed_state": self._speed_state[actor_id],
            "rl_switch": self._rl_switch[actor_id],
        }})

        return self._get_state(actor_id)
        
    def _speed_switch(self, actor_id, cont):
        """cont: the control command of RL agent"""
        ego_speed = get_speed(self._actors[actor_id])
        if self._speed_state[actor_id] == SpeedState.START:
            hero_autopilot(self._actors[actor_id], self._traffic_manager, 
                           self._actor_configs[actor_id], self._env_config, True)
            # control = self.controller.run_step({'waypoints':self.next_wps,'vehicle_front':self.vehicle_front})
            if ego_speed >= self._actor_configs[actor_id]["speed_threshold"]:
                self._speed_state[actor_id] = SpeedState.RUNNING
                if self._actor_configs[actor_id]["auto_control"]:
                    hero_autopilot(self._actors[actor_id], self._traffic_manager, 
                                self._actor_configs[actor_id], self._env_config, True)
                else:
                    hero_autopilot(self._actors[actor_id], self._traffic_manager, 
                                self._actor_configs[actor_id], self._env_config, False)
                    if self._rl_switch[actor_id]:
                        # RL in control
                        pass
                    else:
                        # PID in control
                        pass
                control = cont
            else:
                control = cont
        elif self._speed_state[actor_id] == SpeedState.RUNNING:
            if self._rl_switch[actor_id]:
                # RL in control
                pass
            else:
                # PID in control
                pass
            control = cont
            if self._state["wps"][actor_id].center_front_wps[2].road_id == self._scenario_map["dest_road_id"]:          
                #Vehicle reaches destination, stop vehicle
                self._speed_state[actor_id] = SpeedState.STOP
                self._collisions[actor_id].stop()
                self._lane_invasions[actor_id].stop()
                hero_autopilot(self._actors[actor_id], self._traffic_manager, self._actor_configs[actor_id],
                           self._env_config, False)
                control = cont
        elif self._speed_state[actor_id] == SpeedState.STOP:
            #Hero vehicle reaches destination, properly stop hero vehicle
            control = cont
        else:
            logging.error('CODE LOGIC ERROR')

        return control

    def _get_state(self, actor_id):
        """return a tuple: the first element is next waypoints, 
        the second element is vehicle_front information"""
        wps_info, lights_info, vehs_info = self._local_planner[actor_id].run_step()
        self._state["wps"].update({actor_id: wps_info})
        self._state["lights"].update({actor_id: lights_info})
        self._state["vehs"].update({actor_id: vehs_info})
        if self._rl_configs["debug"]:
            draw_waypoints(self.world, wps_info.center_front_wps+wps_info.center_rear_wps+\
                wps_info.left_front_wps+wps_info.left_rear_wps+wps_info.right_front_wps+wps_info.right_rear_wps, 
                self._fixed_delta_seconds + 0.001, z=1, 
                color=random.choice([(255,0,0)]))

        left_wps=wps_info.left_front_wps
        center_wps=wps_info.center_front_wps
        right_wps=wps_info.right_front_wps

        lane_center = get_lane_center(self.map, self._actors[actor_id].get_location())
        right_lane_dis = lane_center.get_right_lane(
            ).transform.location.distance(self._actors[actor_id].get_location())
        ego_t= lane_center.lane_width / 2 + lane_center.get_right_lane().lane_width / 2 - right_lane_dis

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
        vehicle_inlane_processed = process_veh(self._actors[actor_id
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
        self._state["meas"][actor_id].update({'velocity': v_s, 'cur_acc': a_s, 
            'prev_acc':self._prev_measurement[actor_id]['cur_acc'] if len(self._prev_measurement)!=0 else 0})
        #update informatino for rear vehicle
        if vehs_info.center_rear_veh is None or \
                (lights_info is not None and lights_info.state!=carla.TrafficLightState.Green):
            self._state["meas"][actor_id].update({
                'rear_id':-1, 'rear_v':0, 'rear_a':0, 'change_lane': None})
        else:
            lane_center=get_lane_center(self.map, vehs_info.center_rear_veh.get_location())
            yaw_forward=lane_center.transform.get_forward_vector()
            v_3d=vehs_info.center_rear_veh.get_velocity()
            v_s,v_t=get_projection(v_3d,yaw_forward)
            a_3d=vehs_info.center_rear_veh.get_acceleration()
            a_s,a_t=get_projection(a_3d,yaw_forward)
            self._state["meas"][actor_id].update({'rear_id':vehs_info.center_rear_veh.id, 
                'rear_v':v_s,'rear_a':a_s, 'change_lane': None})

        return {'left_waypoints': left_wps_processed, 'center_waypoints': center_wps_processed,
                'right_waypoints': right_wps_processed, 'vehicle_info': vehicle_inlane_processed,
                'hero_vehicle': [v_s/10, v_t/10, a_s/3, a_t/3, ego_t, yaw_diff_ego/90],
                'light':light}  
  
    def _truncated(self, actor_id):
        """Decide whether terminating current episode or not"""
        m = self._state["meas"][actor_id]
        lane_center=get_lane_center(self.map, self._actors[actor_id].get_location())
        yaw_diff = math.degrees(get_yaw_diff(lane_center.transform.get_forward_vector(),
                        self._actors[actor_id].get_transform().get_forward_vector()))

        if (m["collision_vehicles"] > 0 or m["collision_pedestrians"]>0 or m["collision_other"] > 0) \
                and self._actor_configs[actor_id].get("early_terminate_on_collision", True):
            # Here we judge speed state because there might be collision event when spawning vehicles
            logging.warn('collison happend')
            return Truncated.COLLISION
        if not test_waypoint(lane_center,False):
            logging.warn('vehicle drive out of road')
            return Truncated.OUT_OF_ROAD
        # if self.current_action == Action.LANE_FOLLOW and self.current_lane != self.last_lane:
        #     logging.warn('change lane in lane following mode')
        #     return Truncated.CHANGE_LANE_IN_LANE_FOLLOW
        # if self.current_action == Action.LANE_CHANGE_LEFT and self.current_lane-self.last_lane<0:
        #     logging.warn('vehicle change to wrong lane')
        #     return Truncated.CHANGE_TO_WRONG_LANE
        # if self.current_action == Action.LANE_CHANGE_RIGHT and self.current_lane-self.last_lane>0:
        #     logging.warn('vehicle change to wrong lane')
        #     return Truncated.CHANGE_TO_WRONG_LANE
        # if self.speed_state!=SpeedState.START and not self.vehs_info.center_front_veh:
        #     if not self.lights_info or self.lights_info.state!=carla.TrafficLightState.Red:
        #         if len(self.vel_buffer)==self.vel_buffer.maxlen:
        #             avg_vel=0
        #             for vel in self.vel_buffer:
        #                 avg_vel+=vel/self.vel_buffer.maxlen
        #             if avg_vel*3.6<self.speed_min:
        #                 logging.warn('vehicle speed too low')
        #                 return Truncated.SPEED_LOW
            
        # if self.lane_invasion_sensor.get_invasion_count()!=0:
        #     logging.warn('lane invasion occur')
        #     return True
        # if self.step_info['Lane_center'] <=-1.0:
        #     logging.warn('drive out of road, lane invasion occur')
        #     return True
        if abs(yaw_diff)>90:
            logging.warn('moving in opposite direction')
            return Truncated.OPPOSITE_DIRECTION
        # if self.lights_info and self.lights_info.state!=carla.TrafficLightState.Green:
        #     self.world.debug.draw_point(self.lights_info.get_location(),size=0.3,life_time=0)
        #     wps=self.lights_info.get_stop_waypoints()
        #     for wp in wps:
        #         self.world.debug.draw_point(wp.transform.location,size=0.1,life_time=0)
        #         if is_within_distance_ahead(self.ego_vehicle.get_location(),wp.transform.location, wp.transform, self.min_distance):
        #             logging.warn('break traffic light rule')
        #             return Truncated.TRAFFIC_LIGHT_BREAK

        return Truncated.FALSE

    def _done(self, actor_id, truncated):
        if truncated!=Truncated.FALSE:
            return False
        if self._speed_state[actor_id] == SpeedState.STOP:
            logging.info('vehicle reach destination, stop '+str(actor_id))                              
            return True
        if not self._rl_switch[actor_id]:
            if self._num_steps[actor_id] > self._scenario_config["max_steps"]:
                # Let the traffic manager only execute 5000 steps. or it can fill the replay buffer
                logging.info(f"{self._scenario_config['max_steps']} "
                             f"steps passed under traffic manager control")
                return True

        return False

    def close(self):
        """Clean-up the world, clear server state & close the Env"""
        self._clean_world()
        self._clear_server_state()


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
            print(
                ":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                    i, reward, total_reward_dict, done
                )
            )

        print("{} fps".format(i / (time.time() - start)))
