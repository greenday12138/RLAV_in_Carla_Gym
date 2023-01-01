import time
import carla
import random
import logging
import pygame
import math, copy
import numpy as np
from enum import Enum
from queue import Queue
from collections import deque
from gym_carla.multi_agent.util.render import World,HUD
from gym_carla.multi_agent.assets.ego_client import EgoClient
from gym_carla.multi_agent.agent.global_planner import GlobalPlanner
from gym_carla.multi_agent.util.misc import draw_waypoints, get_speed, get_acceleration, test_waypoint, \
    compute_distance, get_actor_polygons, get_lane_center, remove_unnecessary_objects, get_yaw_diff, \
    get_trafficlight_trigger_location, is_within_distance, get_sign,is_within_distance_ahead,get_projection,\
    create_vehicle_blueprint

class CarlaEnv:
    def __init__(self, args) -> None:
        super().__init__()
        self.host = args.host
        self.port = args.port
        self.tm_port = args.tm_port
        self.sync = args.sync
        self.fps = args.fps
        self.hybrid = args.hybrid
        self.no_rendering = args.no_rendering
        self.speed_limit = args.speed_limit
        self.num_of_vehicles = args.num_of_vehicles
        self.sampling_resolution = args.sampling_resolution
        self.ignore_traffic_light = args.ignore_traffic_light

        logging.info('listening to server %s:%s', args.host, args.port)
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.sim_world = self.client.load_world(args.map)
        remove_unnecessary_objects(self.sim_world)
        self.map = self.sim_world.get_map()
        self.origin_settings = self.sim_world.get_settings()
        self.traffic_manager = None
        # Set fixed simulation step for synchronous mode
        self._set_synchronous_mode()
        self._set_traffic_manager()
        logging.info('Carla server connected')

        #init pygame window
        self.pygame=args.pygame and not self.no_rendering
        self.width,self.height=[int(x) for x in args.res.split('x')]
        if self.pygame :
            self._init_renderer()

        # generate ego vehicle spawn points on chosen route
        self.global_planner = GlobalPlanner(self.map, self.sampling_resolution)
        self.spawn_points = self.global_planner.get_spawn_points()
        # arguments for debug
        self.debug = args.debug
        self.seed = args.seed
        if self.debug:
            # draw_waypoints(self.sim_world,self.global_panner.get_route())
            random.seed(self.seed)
        self.rl_control_step=0

        # Set weather
        # self.sim_world.set_weather(carla.WeatherParamertes.ClearNoon)

        self.companion_vehicles = []
        self.ego_num=args.ego_num
        self.ego_clients=[
            EgoClient(args,{'Client':self.client,'ID':i,'Spawn_points':self.spawn_points}) for i in range(self.ego_num)]

        # thread blocker
        self.sensor_queue = Queue(maxsize=10)
        self.camera = None

    def __del__(self):
        logging.info('\n Destroying all vehicles')
        self.sim_world.apply_settings(self.origin_settings)
        self._clear_actors(['vehicle.*', 'sensor.other.collison', 'sensor.camera.rgb', 'sensor.other.lane_invasion'])

    def reset(self):
        for ego in self.ego_clients:
            ego.reset_before_tick()
        if len(self.companion_vehicles)!=0:
            if self.camera is not None:
                self.camera.stop()
            self._clear_actors(
                ['*vehicle.*', 'sensor.other.collision', 'sensor.camera.rgb', 'sensor.other.lane_invasion'])
            self.companion_vehicles.clear()
            if self.pygame:
                self.world.destroy()
                pygame.quit()

        # Spawn surrounding vehicles
        self._spawn_companion_vehicles()
        #set traffic light elpse time
        lights_list=self.sim_world.get_actors().filter("*traffic_light*")
        for light in lights_list:
            light.set_green_time(10)
            light.set_red_time(5)
            light.set_yellow_time(0)

        # friction_bp=self.sim_world.get_blueprint_library().find('static.trigger.friction')
        # bb_extent=self.ego_vehicle.bounding_box.extent
        # friction_bp.set_attribute('friction',str(0.0))
        # friction_bp.set_attribute('extent_x',str(bb_extent.x))
        # friction_bp.set_attribute('extent_y',str(bb_extent.y))
        # friction_bp.set_attribute('extent_z',str(bb_extent.z))
        # self.sim_world.spawn_actor(friction_bp,self.ego_vehicle.get_transform())
        # self.sim_world.debug.draw_box()

        # let the client interact with server
        if self.sync:
            if self.pygame:
                self._init_renderer()
                self.world.restart(self.ego_clients[0].ego_vehicle)
            else:
                self.sim_world.tick()

                # spectator = self.sim_world.get_spectator()
                # transform = self.ego_vehicle.get_transform()
                # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=100),
                #                                         carla.Rotation(pitch=-90)))

            # code for synchronous mode
            # camera_bp = self.sim_world.get_blueprint_library().find('sensor.camera.rgb')
            # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            # self.camera = self.sim_world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_clients[0].ego_vehicle)
            # self.camera.listen(lambda image: self._sensor_callback(image, self.sensor_queue))
        else:
            self.sim_world.wait_for_tick()

        states=[]
        for ego in self.ego_clients:
            states.append(ego.reset_after_tick())

        # return state information
        return  states

    def step(self, a_indexs, actions):
        self.all_under_rl_control=True
        if self.sync:
            for i,ego in enumerate(self.ego_clients):
                ego.step_before_tick(a_indexs[i],actions[i])

            # print(self.map.get_waypoint(self.ego_vehicle.get_location(),False),self.ego_vehicle.get_transform(),sep='\n')
            # print(self.sim_world.get_snapshot().timestamp)
            if self.pygame:
                self._tick()
            else:
                self.sim_world.tick()
                # spectator = self.sim_world.get_spectator()
                # transform = self.ego_clients[0].ego_vehicle.get_transform()
                # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
                #                                         carla.Rotation(pitch=-90)))

            # camera_data = self.sensor_queue.get(block=True)
            """Attention: the server's tick function only returns after it ran a fixed_delta_seconds, so the client need not to wait for
            the server, the world snapshot of tick returned already include the next state after the uploaded action."""
            # print(self.map.get_waypoint(self.ego_vehicle.get_location(),False),self.ego_vehicle.get_transform(),sep='\n')
            # print(self.sim_world.get_snapshot().timestamp)
            # print()
            states,rewards,truncateds,dones,infos=[],[],[],[],[]
            for ego in self.ego_clients:
                s,r,t,d,i=ego.step_after_tick()
                states.append(s)
                rewards.append(r)
                truncateds.append(t)
                dones.append(d)
                infos.append(i)

                if not ego.is_effective_action() or not ego.RL_switch==True:
                    self.all_under_rl_control=False

            if self.all_under_rl_control:
                self.rl_control_step+=1
        else:
            temp = self.sim_world.wait_for_tick()
            self.sim_world.on_tick(lambda _: {})
            time.sleep(1.0 / self.fps)
            states,rewards,truncateds,dones,infos=[],[],[],[],[]

        return states,rewards,truncateds,dones,infos

    def get_observation_space(self):
        """
        :return:
        """
        """Get observation space of cureent environment"""
        return self.ego_clients[0].get_observation_space()

    def get_action_bound(self):
        """Return action bound of ego vehicle controller"""
        return self.ego_clients[0].get_action_bound()

    def is_effective_action(self,id=None):
        """id !=None: testing if ith ego vehcle's action should be put into replay buffer(under rl control)
            id ==None: testing if all ego vehicles are under rl control"""
        if id is not None:
            return self.ego_clients[id].is_effective_action()
        else:
            return self.all_under_rl_control

    def seed(self, seed=None):
        return

    def render(self, mode):
        pass

    def _sensor_callback(self, sensor_data, sensor_queue):
        array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('uint8'))
        # image is rgba format
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
        array = array[:, :, :3]
        sensor_queue.put((sensor_data.frame, array))

    def _tick(self):
        self.clock.tick()
        #self.sim_world.tick()
        if self.sync:
            self.world.world.tick()
        else:
            self.world.world.wait_for_tick()
        self.world.tick(self.clock)
        self.world.render(self.display)
        pygame.display.flip()

    def _init_renderer(self):
        """Initialize the birdeye view renderer."""
        pygame.init()
        pygame.font.init()
        self.display=pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.hud=HUD(self.width,self.height)
        self.world=World(self.sim_world,self.hud)
        self.clock=pygame.time.Clock()

    def _set_synchronous_mode(self):
        """Set whether to use the synchronous mode."""
        # Set fixed simulation step for synchronous mode
        if self.sync:
            settings = self.sim_world.get_settings()
            settings.no_rendering_mode = self.no_rendering
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.fps
                self.sim_world.apply_settings(settings)

    def _set_traffic_manager(self):
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        # every vehicle keeps a distance of 3.0 meter
        self.traffic_manager.set_global_distance_to_leading_vehicle(10)
        # Set physical mode only for cars around ego vehicle to save computation
        if self.hybrid:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(200.0)

        """The default global speed limit is 30 m/s
        Vehicles' target speed is 70% of their current speed limit unless any other value is set."""
        speed_diff = (30 * 3.6 - (self.speed_limit+1)) / (30 * 3.6) * 100
        # Let the companion vehicles drive a bit faster than ego speed limit
        self.traffic_manager.global_percentage_speed_difference(-100)
        self.traffic_manager.set_synchronous_mode(self.sync)
        #set traffic light elpse time
        lights_list=self.sim_world.get_actors().filter("*traffic_light*")
        for light in lights_list:
            light.set_green_time(15)
            light.set_red_time(0)
            light.set_yellow_time(0)

    def _spawn_companion_vehicles(self):
        """
        Spawn surrounding vehcles of this simulation
        each vehicle is set to autopilot mode and controled by Traffic Maneger
        note: the ego vehicle trafficmanager and companion vehicle trafficmanager shouldn't be the same one
        """
        # spawn_points_ = self.map.get_spawn_points()
        spawn_points_ = self.spawn_points
        # make sure companion vehicles also spawn on chosen route
        # spawn_points_=[x.transform for x in self.ego_spawn_waypoints]

        num_of_spawn_points = len(spawn_points_)
        num_of_vehicles=random.choice(self.num_of_vehicles)

        if num_of_vehicles < num_of_spawn_points:
            random.shuffle(spawn_points_)
            spawn_points = random.sample(spawn_points_, num_of_vehicles)
        else:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_of_vehicles, num_of_spawn_points)
            num_of_vehicles = num_of_spawn_points - 1

        # Use command to apply actions on batch of data
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor  # FutureActor is eaqual to 0
        command_batch = []

        for i, transform in enumerate(spawn_points_):
            if i >= num_of_vehicles:
                break

            blueprint = create_vehicle_blueprint(self.sim_world,'vehicle.audi.etron',ego=False,color='0,0,0',number_of_wheels=[4])
            # Spawn the cars and their autopilot all together
            command_batch.append(SpawnActor(blueprint, transform).
                                 then(SetAutopilot(FutureActor, True, self.tm_port)))

        # execute the command batch
        for (i, response) in enumerate(self.client.apply_batch_sync(command_batch, self.sync)):
            if response.has_error():
                logging.warn(response.error)
            else:
                # print("Future Actor",response.actor_id)
                vehicle=self.sim_world.get_actor(response.actor_id)
                self.companion_vehicles.append(vehicle)
                
                if self.ignore_traffic_light:
                    self.traffic_manager.ignore_lights_percentage(vehicle, 100)
                    self.traffic_manager.ignore_walkers_percentage(vehicle, 100)
                else:
                    self.traffic_manager.ignore_lights_percentage(vehicle, 50)
                    self.traffic_manager.ignore_walkers_percentage(vehicle, 50)
                self.traffic_manager.ignore_signs_percentage(vehicle, 100)
                self.traffic_manager.auto_lane_change(vehicle, True)
                # modify change probability
                self.traffic_manager.random_left_lanechange_percentage(vehicle, 0)
                self.traffic_manager.random_right_lanechange_percentage(vehicle, 0)
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle,
                        random.choice([30,20,10,0,-20-40,-60,-80,-100,-100,-100]))
                self.traffic_manager.set_route(vehicle,
                                               ['Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight'])
                self.traffic_manager.update_vehicle_lights(vehicle, True)
                # print(vehicle.attributes)

        msg = 'requested %d vehicles, generate %d vehicles, press Ctrl+C to exit.'
        logging.info(msg, num_of_vehicles, len(self.companion_vehicles))

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        pass

    def _clear_actors(self, actor_filters, filter=True):
        """Clear specific actors
        filter: True means filter actors by blueprint, Fals means fiter actors by carla.CityObjectLabel"""
        if filter:
            for actor_filter in actor_filters:
                self.client.apply_batch([carla.command.DestroyActor(x)
                                         for x in self.sim_world.get_actors().filter(actor_filter)])

        # for actor_filter in actor_filters:
        #     for actor in self.sim_world.get_actors().filter(actor_filter):
        #         if actor.is_alive:
        #             if actor.type_id =='controller.ai.walker':
        #                 actor.stop()
        #             actor.destroy()
