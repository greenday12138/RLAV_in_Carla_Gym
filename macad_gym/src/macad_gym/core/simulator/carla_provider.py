import carla
import time
import socket
import GPUtil
import random
import shutil
import subprocess
import traceback
import os, sys
import signal
import psutil
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from macad_gym import IS_WINDOWS_PLATFORM, SERVER_BINARY
from macad_gym.viz.logger import LOG


def termination_cleanup(*_):
    CarlaConnector.clean_up()
    sys.exit(0)

class CarlaError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CarlaConnector(object):
    live_carla_processes = set()

    def __init__(self, server_map, env_config) -> None:
        super().__init__()
        self._client = None
        self._world = None
        self._map = None
        self._traffic_manager = None
        self.server_pid = None

        # Create a new server process and start the client.
        # self._connected_carla.append(CarlaConnector(self._server_port, self._server_map, self._env_config))
        # self._carla = weakref.proxy(self._connected_carla[-1])
        self._server_port, server_process =  CarlaConnector.connect(LOG.multi_env_logger, env_config)
        self.server_pid = server_process.pid

        if IS_WINDOWS_PLATFORM:
            self.server_pid = server_process.pid
        else:
            # The carla setup procedure start a process group in Linux.
            self.server_pid = os.getpgid(server_process.pid)

        CarlaConnector.live_carla_processes.add(self.server_pid)

        while self._client is None:
            try:
                self._client = carla.Client("localhost", self._server_port, worker_threads=0)
                # The socket establishment could takes some time
                time.sleep(2)
                self._client.set_timeout(10.0)
                LOG.multi_env_logger.info(
                    f"Client successfully connected to server, Live Carla Process ID:{CarlaConnector.live_carla_processes}\n"
                    f"Carla-Server version: {self._client.get_server_version()}, Carla-Client version:{self._client.get_client_version()}")
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    LOG.multi_env_logger.error(f"Could not connect to Carla server because:{re}")
                self._client = None

        # load map using client api since 0.9.6+
        self._client.load_world(server_map, reset_settings=False, 
                                map_layers=carla.MapLayer.NONE)
        self._world = self._client.get_world()
        self._map = self._world.get_map()
        #remove_unnecessary_objects(self._world)
        
        # Sign on traffic manager
        self._traffic_manager = self._client.get_trafficmanager(self._server_port + 4)
        # Actors will become dormant 2km away from here vehicle
        world_settings = self._world.get_settings()
        if env_config["hybrid"]:
            world_settings.actor_active_distance = 2000 
        world_settings.synchronous_mode = env_config["sync_server"]
        if env_config["sync_server"]:
            # Synchronous mode
            # try:
            # Available with CARLA version>=0.9.6
            # Set fixed_delta_seconds to have reliable physics between sim steps
            world_settings.fixed_delta_seconds = env_config["fixed_delta_seconds"]
            self._traffic_manager.set_synchronous_mode(True)
        self._world.apply_settings(world_settings)
        self.tick(LOG.multi_env_logger)

    def disconnect(self):
        # XXX: traffic_manager.shut_donw() is critical in carla server restart procedure.
        # Without such code, you cannot clear local client cache and world cache, so you'll still be connecting to previous carla server port 
        # even if you start another carla server and call carla.Client("localhost", new_carla_port).
        # This bug cost me half month……
        self._traffic_manager.shut_down()
        del self._traffic_manager
        del self._map
        del self._world
        del self._client
        self._client, self._map, self._world, self._traffic_manager = None, None, None, None

        if self.server_pid and psutil.pid_exists(self.server_pid):
            if IS_WINDOWS_PLATFORM:
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(self.server_pid)])
            else:
                    os.killpg(self.server_pid, signal.SIGKILL)

            CarlaConnector.live_carla_processes.remove(self.server_pid)

        self._server_port = None
        self.server_pid = None

    def set_weather(self, weather, logger):
        try:
            self._world.set_weather(weather)
        except RuntimeError as e:
            logger.exception("Carla world set_weather failed, restart carla!")
            raise CarlaError(e.args) from e

    def get_weather(self, logger):
        weas = None
        try:
            weas = self._world.get_weather()
        except RuntimeError as e:
            logger.exception("Carla world get_weather failed, restart carla!")
            raise CarlaError(e.args) from e
        
        return weas

    def get_traffic_light(self, logger):
        tras = None
        try:
            tras = self._world.get_traffic_light()
        except RuntimeError as e:
            logger.exception("Carla world get_traffic_light failed, restart carla!")
            raise CarlaError(e.args) from e
        
        return tras

    def get_spectator(self, logger):
        spe = None
        try:
            spe = self._world.get_spectator()
        except RuntimeError as e:
            logger.exception("Carla world get_spectator failed, restart carla!")
            raise CarlaError(e.args) from e
        
        return spe

    def get_blueprint_library(self, logger):
        lib = None
        try:
            lib = self._world.get_blueprint_library()
        except RuntimeError as e:
            logger.exception("Carla world get_blueprint_library failed, restart carla!")
            raise CarlaError(e.args) from e
        
        return lib

    def tick(self, logger):
        try:
            self._world.tick()
        except RuntimeError as e:
            logger.exception("Carla world tick failed, restart carla!")
            raise CarlaError(e.args) from e

    @staticmethod
    def clean_up():
        LOG.multi_env_logger.info(f"Killing live carla processes:{CarlaConnector.live_carla_processes}")
        for pgid in CarlaConnector.live_carla_processes:
            if IS_WINDOWS_PLATFORM:
                # for Windows
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(pgid)])
            else:
                # for Linux
                os.killpg(pgid, signal.SIGKILL)

        CarlaConnector.live_carla_processes.clear()

    @staticmethod
    def get_tcp_port(port=0):
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

    @staticmethod
    def connect(logger, env_config):
        # First find a port that is free and then use it in order to avoid
        # crashes due to:"...bind:Address already in use"
        server_port = CarlaConnector.get_tcp_port()
        logger.info(
            f"1. Port: {server_port}\n"
            f"2. Map: {env_config['server_map']}\n"
            f"3. Binary: {SERVER_BINARY}"
        )
        multigpu_success = False
        server_process = None
        gpus = GPUtil.getGPUs()
        if not env_config["render"] and (gpus is not None and len(gpus)) > 0:
            try:
                min_index = random.randint(0, len(gpus) - 1)
                for i, gpu in enumerate(gpus):
                    if gpu.load < gpus[min_index].load:
                        min_index = i
                # Check if vglrun is setup to launch sim on multipl GPUs
                if shutil.which("vglrun") is not None:
                    server_process = subprocess.Popen(
                        (
                            "DISPLAY=:8 vglrun -d :7.{} {} -benchmark -fps={}"
                            " -carla-server -world-port={}"
                            " -carla-streaming-port=0".format(
                                min_index,
                                SERVER_BINARY,
                                1/env_config["fixed_delta_seconds"],
                                server_port,
                            )
                        ),
                        shell=True,
                        # for Linux
                        preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                        # for Windows (not necessary)
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                        if IS_WINDOWS_PLATFORM
                        else 0,
                        stdout=open(LOG.server_log, 'w'),
                    )

                # Else, run in headless mode
                # else:
                #     # Since carla 0.9.12+ use -RenderOffScreen to start headlessly
                #     # https://carla.readthedocs.io/en/latest/adv_rendering_options/
                #     server_process = subprocess.Popen(
                #         (  # 'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE={} DISPLAY='
                #             '"{}" -RenderOffScreen -benchmark -fps={} -carla-server'
                #             " -world-port={} -carla-streaming-port=0".format(
                #                 SERVER_BINARY,
                #                 1/env_config["fixed_delta_seconds"],
                #                 server_port,
                #             )
                #         ),
                #         shell=True,
                #         # for Linux
                #         preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                #         # for Windows (not necessary)
                #         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                #         if IS_WINDOWS_PLATFORM
                #         else 0,
                #         stdout=open(LOG.server_log, 'w'),
                #     )
            # TODO: Make the try-except style handling work with Popen
            # exceptions after launching the server procs are not caught
            except Exception as e:
                logger.exception(traceback.format_exc())
            # Temporary soln to check if CARLA server proc started and wrote
            # something to stdout which is the usual case during startup
            if server_process is None:
                multigpu_success = False
            else:
                multigpu_success = True

            if multigpu_success:
                logger.info("Running sim servers in headless/multi-GPU mode")

        # Rendering mode and also a fallback if headless/multi-GPU doesn't work
        if multigpu_success is False:
            try:
                logger.info("Using single gpu to initialize carla server")
                parameters = [SERVER_BINARY, 
                              "-windowed", 
                              #"-prefernvidia",
                              "-quality-level=Low",
                              f"-ResX={str(env_config['render_x_res'])}",
                              f"-ResY={str(env_config['render_y_res'])}",
                              "-benchmark",
                              "-fps={}".format(1/env_config["fixed_delta_seconds"]),
                              "-carla-server",
                              "-carla-rpc-port={}".format(server_port),
                              "-carla-streaming-port=0"
                              "-StompMAlloc"]
                if not env_config["render"]:
                    parameters.append("-RenderOffScreen")

                server_process = subprocess.Popen(
                    parameters,
                    # for Linux
                    preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                    # for Windows (not necessary)
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    if IS_WINDOWS_PLATFORM
                    else 0,
                    stdout=open(LOG.server_log, 'w'),
                    #bufsize=131072
                )
                logger.info("Running simulation in single-GPU mode")
            except Exception as e:
                logger.error(f"FATAL ERROR while launching server:{sys.exc_info()[0]}")
            
        return server_port, server_process



class CarlaDataProvider(object):
    """
    This class provides access to various data for all registered actors
    In addition it provides access to the map and the transform of all traffic lights
    It buffers the data and updates it on every CARLA tick
    It aims to get rid of frequently updating data from Carla server

    """

    _client = None
    _world = None
    _map = None
    _sync_mode = True
    _actor_speed_map = dict()
    _actor_transform_map = dict()
    _actor_acceleration_map = dict()
    _actor_angular_velocity_map = dict()
    _actor_speed_vector_map = dict()
    _traffic_light_map = dict()
    _carla_actor_pool = dict()
    _hero_vehicle_route = None
    _target_waypoint = None
    _spawn_points = None
    _available_points = set()
    _blueprint_library = None
    _rng = np.random.RandomState(2000)