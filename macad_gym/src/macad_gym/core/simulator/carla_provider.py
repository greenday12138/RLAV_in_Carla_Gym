import carla
import socket
import GPUtil
import random
import shutil
import subprocess
import traceback
import os, sys
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from macad_gym import IS_WINDOWS_PLATFORM, SERVER_BINARY
from macad_gym.viz.logger import LOG


class CarlaError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CarlaConnector(object):
    @staticmethod
    def set_weather(world, weather, logger):
        try:
            world.set_weather(weather)
        except RuntimeError as e:
            logger.exception("Carla world set_weather failed, restart carla!")
            raise CarlaError(e.args) from e

    @staticmethod
    def get_weather(world, logger):
        weas = None
        try:
            weas = world.get_weather()
        except RuntimeError as e:
            logger.exception("Carla world get_weather failed, restart carla!")
            raise CarlaError(e.args) from e
        
        return weas

    @staticmethod
    def get_traffic_light(world, logger):
        tras = None
        try:
            tras = world.get_traffic_light()
        except RuntimeError as e:
            logger.exception("Carla world get_traffic_light failed, restart carla!")
            raise CarlaError(e.args) from e
        
        return tras

    @staticmethod
    def get_spectator(world, logger):
        spe = None
        try:
            spe = world.get_spectator()
        except RuntimeError as e:
            logger.exception("Carla world get_spectator failed, restart carla!")
            raise CarlaError(e.args) from e
        
        return spe

    @staticmethod
    def get_blueprint_library(world, logger):
        lib = None
        try:
            lib = world.get_blueprint_library()
        except RuntimeError as e:
            logger.exception("Carla world get_blueprint_library failed, restart carla!")
            raise CarlaError(e.args) from e
        
        return lib

    @staticmethod
    def tick(world, logger):
        try:
            world.tick()
        except RuntimeError as e:
            logger.exception("Carla world tick failed, restart carla!")
            raise CarlaError(e.args) from e

    @staticmethod
    def clear_server(process):
        pass

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
                              "-carla-streaming-port=0"]
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

    pass