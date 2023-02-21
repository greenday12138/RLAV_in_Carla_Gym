import logging
import carla
import random
import math, time
import numpy as np
import time,pygame
from enum import Enum
from queue import Queue
#from gym_carla.env.agent.basic_agent import BasicAgent
from gym_carla.single_lane.util.render import World,HUD
from gym_carla.single_lane.util.misc import draw_waypoints, get_speed, get_acceleration, test_waypoint, \
    compute_distance, get_actor_polygons, get_lane_center, remove_unnecessary_objects,get_yaw_diff,create_vehicle_blueprint
from gym_carla.single_lane.util.sensor import CollisionSensor, LaneInvasionSensor, SemanticTags
from gym_carla.single_lane.agent.local_planner import LocalPlanner
from gym_carla.single_lane.agent.global_planner import GlobalPlanner
from gym_carla.single_lane.agent.pid_controller import VehiclePIDController
from gym_carla.single_lane.carla.behavior_agent import BehaviorAgent,BasicAgent

class SpeedState(Enum):
    """Different ego vehicle speed state
        START: Initializing state, speed up the vehicle to speed_threshole, use basic agent controller
        RUNNING: After initializing, ego speed between speed_min and speed_limit, use RL controller
        REBOOT: After initializaing, ego speed reaches below speed min, use basic agent controller to speed up ego vehicle to speed_threshold
    """
    START = 0
    RUNNING = 1
    RUNNING_RL = 2
    RUNNING_PID = 3
    REBOOT = 4

class ControlInfo:
    """Wrapper for vehicle(model3) control info"""
    def __init__(self,throttle=0.0,brake=0.0,steer=0.0,exec_steps=1,gear=1,) -> None:
        self.throttle=throttle
        self.steer=steer
        self.brake=brake
        self.gear=gear
        self.reverse=False
        self.manual_gear_shift=False
        self.exec_steps=exec_steps
        self.exec_steps_info=-2

class CarlaEnv:
    def __init__(self, args) -> None:
        super().__init__()
        self.host = args.host
        self.port = args.port
        self.tm_port = args.tm_port
        self.sync = args.sync
        self.fps = args.fps
        self.no_rendering = args.no_rendering
        self.ego_filter = args.filter
        self.loop = args.loop
        self.agent = args.agent
        # arguments for debug
        self.debug = args.debug
        self.train = args.train  # argument indicating training agent
        self.adapt = args.adapt
        self.seed = args.seed
        self.behavior = args.behavior
        self.res = args.res
        self.num_of_vehicles = args.num_of_vehicles
        self.sampling_resolution = args.sampling_resolution
        self.min_distance=args.min_distance
        self.vehicle_proximity = args.vehicle_proximity
        self.hybrid = args.hybrid
        self.stride = args.stride
        self.buffer_size = args.buffer_size
        if self.train:
            self.pre_train_steps = args.pre_train_steps
        else:
            self.pre_train_steps= 0
        self.speed_limit = args.speed_limit
        # The RL agent acts only after ego vehicle speed reach speed threshold
        self.speed_threshold = args.speed_threshold
        self.speed_min = args.speed_min
        # controller action space
        self.steer_bound = args.steer_bound
        self.throttle_bound = args.throttle_bound
        self.brake_bound = args.brake_bound

        logging.info('listening to server %s:%s', args.host, args.port)
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.sim_world = self.client.load_world(args.map)
        remove_unnecessary_objects(self.sim_world)
        self.map = self.sim_world.get_map()
        self.origin_settings = self.sim_world.get_settings()
        self.traffic_manager = None
        self.speed_state = SpeedState.START
        # Set fixed simulation step for synchronous mode
        self._set_synchronous_mode()
        self._set_traffic_manager()
        # Set weather
        # self.sim_world.set_weather(carla.WeatherParamertes.ClearNoon)
        logging.info('Carla server connected')

        # Record the time of total steps
        self.reset_step = 0
        self.total_step = 0
        self.time_step = 0
        self.rl_control_step = 0
        self.switch_count=0
        # Let the RL controller and PID controller alternatively take control every 500 steps
        # RL_switch: True--currently RL in control, False--currently PID in control
        self.RL_switch = False
        self.SWITCH_THRESHOLD = args.switch_threshold
        self.next_wps = None  # ego vehicle's following waypoint list

        # generate ego vehicle spawn points on chosen route
        self.global_planner = GlobalPlanner(self.map, self.sampling_resolution)
        self.local_planner = None
        self.spawn_points = self.global_planner.get_spawn_points()
        self.ego_spawn_point = None
        # former_wp record the ego vehicle waypoint of former step
        self.former_wp = None

        # arguments for caculating reward
        self.TTC_THRESHOLD = args.TTC_th
        self.penalty = args.penalty
        self.last_acc = carla.Vector3D()  # ego vehicle acceration in last step
        self.step_info = None
        self.control= ControlInfo()

        if self.debug:
            # draw_waypoints(self.sim_world,self.global_panner.get_route())
            random.seed(self.seed)

        self.companion_vehicles = []
        self.vehicle_polygons = []
        self.ego_vehicle = None
        # the vehicle in front of ego vehicle
        self.vehicle_front = None

        # Collision sensor
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        #init pygame window
        self.pygame=args.pygame and not self.no_rendering
        self.width,self.height=[int(x) for x in args.res.split('x')]
        if self.pygame :
            self._init_renderer()

        # thread blocker
        self.sensor_queue = Queue(maxsize=10)
        self.camera = None

    def __del__(self):
        logging.info('\n Destroying all vehicles')
        self.sim_world.apply_settings(self.origin_settings)
        self._clear_actors(['vehicle.*', 'sensor.other.collison', 'sensor.camera.rgb', 'sensor.other.lane_invasion'])

    def reset(self):
        if self.ego_vehicle is not None:
            # self.sim_world.apply_settings(self.origin_settings)
            # self._set_synchronous_mode()
            self._clear_actors(
                ['*vehicle.*', 'sensor.other.collison', 'sensor.camera.rgb', 'sensor.other.lane_invasion'])
            self.ego_vehicle = None
            self.vehicle_polygons.clear()
            self.companion_vehicles.clear()
            self.collision_sensor = None
            self.lane_invasion_sensor = None
            # self.camera = None
            # while (self.sensor_queue.empty() is False):
            #     self.sensor_queue.get(block=False)
            if self.pygame:
                self.world.destroy()
                pygame.quit()

        # Spawn surrounding vehicles
        self._spawn_companion_vehicles()

        # Get actors polygon list
        vehicle_poly_dict = get_actor_polygons(self.sim_world, 'vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)

        # try to spawn ego vehicle
        while self.ego_vehicle is None:
            self.ego_spawn_point = random.choice(self.spawn_points)
            self.former_wp = get_lane_center(self.map, self.ego_spawn_point.location)
            self.ego_vehicle = self._try_spawn_ego_vehicle_at(self.ego_spawn_point)
        # self.ego_vehicle.set_simulate_physics(False)
        self.collision_sensor = CollisionSensor(self.ego_vehicle)
        self.lane_invasion_sensor = LaneInvasionSensor(self.ego_vehicle)

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
                self.world.restart(self.ego_vehicle)
            else:
                spectator = self.sim_world.get_spectator()
                transform = self.ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))
                self.sim_world.tick()
        else:
            self.sim_world.wait_for_tick()

        """Attention:
        get_location() Returns the actor's location the client recieved during last tick. The method does not call the simulator.
        Hence, upon initializing, the world should first tick before calling get_location, or it could cause fatal bug"""
        # self.ego_vehicle.get_location()

        # add route planner for ego vehicle
        self.local_planner = LocalPlanner(self.ego_vehicle, 
            {'sampling_resolution':self.sampling_resolution,
            'buffer_size':self.buffer_size,
            'vehicle_proximity':self.vehicle_proximity})
        # self.local_planner.set_global_plan(self.global_planner.get_route(
        #      self.map.get_waypoint(self.ego_vehicle.get_location())))
        self.next_wps, _, self.vehicle_front = self.local_planner.run_step()

        # set ego vehicle controller
        self._ego_autopilot(True)

        # Only use RL controller after ego vehicle speed reach 10 m/s
        self.speed_state = SpeedState.START
        # self.controller = BasicAgent(self.ego_vehicle, {'target_speed': self.speed_threshold, 'dt': 1 / self.fps,
        #                                                 'max_throttle': self.throttle_bound,
        #                                                 'max_brake': self.brake_bound})
        self.autopilot_controller=BasicAgent(self.ego_vehicle,target_speed=30,
            opt_dict={'ignore_traffic_lights':True,'ignore_stop_signs':True,
            'sampling_resolution':self.sampling_resolution,'dt':1.0/self.fps,
            'sampling_radius':self.sampling_resolution,'max_steering':self.steer_bound,
            'max_throttle':self.throttle_bound,'max_brake':self.brake_bound,
            'ignore_vehicles':random.choice([True,False])})
        # self.control_sigma={'Steer':random.choice([0.3, 0.4, 0.5]),
        #                 'Throttle_brake':random.choice([0.4,0.5,0.6])}
        self.control_sigma={'Steer':random.choice([0,0,0.05,0.1,0.15,0.2,0.25]),
                            'Throttle_brake':random.choice([0,0,0.1,0.2,0.3,0.4,0.5])}

        # code for synchronous mode
        # camera_bp = self.sim_world.get_blueprint_library().find('sensor.camera.rgb')
        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        # self.camera = self.sim_world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        # self.camera.listen(lambda image: self._sensor_callback(image, self.sensor_queue))
        # 

        # speed state switch
        if not self.debug:
            if self.total_step-self.rl_control_step <self.pre_train_steps:
                #During pre-train steps, let rl and pid alternatively take control
                if self.RL_switch:
                    if self.switch_count>=self.SWITCH_THRESHOLD:
                        self.RL_switch=False
                        self.switch_count=0
                    else:
                        self.switch_count+=1
                else:
                    self.RL_switch=True
                    self.switch_count+=1
            else:
                self.RL_switch=True
                # self.local_planner.set_global_plan(self.global_planner.get_route(
                #     self.map.get_waypoint(self.ego_vehicle.get_location())))
        else:
            self.RL_switch=False

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # return state information
        return self._get_state({'waypoints': self.next_wps, 'vehicle_front': self.vehicle_front})

    def step(self, action):
        self.step_info=None
        self.next_wps=None
        #self.vehicle_front=None
        """throttle (float):A scalar value to control the vehicle throttle [0.0, 1.0]. Default is 0.0.
                steer (float):A scalar value to control the vehicle steering [-1.0, 1.0]. Default is 0.0.
                brake (float):A scalar value to control the vehicle brake [0.0, 1.0]. Default is 0.0."""
        steer = np.clip(action[0][0], -self.steer_bound, self.steer_bound)
        if action[0][1] >= 0:
            brake = 0
            throttle = np.clip(action[0][1], 0 ,self.throttle_bound)
        else:
            throttle = 0
            brake = np.clip(abs(action[0][1]), 0 , self.brake_bound)
        if self.adapt:
            if -1<=action[0][2]<-0.5:
                exec_steps=1
            elif -0.5<=action[0][2]<0:
                exec_steps=2
            elif 0<=action[0][2]<0.5:
                exec_steps=3
            elif 0.5<=action[0][2]<=1:
                exec_steps=4
            else:
                logging.warn("EXECUTION STEPS ERROR")
                exec_steps=1
            self.control.exec_steps_info = action[0][2]
        else:
            exec_steps=1

        self.control.throttle, self.control.brake, self.control.steer, self.control.exec_steps=throttle, brake, steer, exec_steps

        # Only use RL controller after ego vehicle speed reach 10 m/s
        # Use DFA to caaulate different speed state transition
        if not self.debug:
            self._speed_switch()
        else:
            self._speed_switch()

        if self.sync:
            if not self.debug:
                if not self.RL_switch:
                    #Add noise to autopilot controller's control command
                    print(f"Basic Agent Control Before Noise:{self.control}")
                    self.control.steer=np.clip(np.random.normal(self.control.steer,self.control_sigma['Steer']),-self.steer_bound,self.steer_bound)
                    if self.control.throttle>0:
                        throttle_brake=self.control.throttle
                    else:
                        throttle_brake=-self.control.brake
                    throttle_brake=np.clip(np.random.normal(throttle_brake,self.control_sigma['Throttle_brake']),-self.brake_bound,self.throttle_bound)
                    if throttle_brake>0:
                        self.control.throttle=throttle_brake
                        self.control.brake=0
                    else:
                        self.control.throttle=0
                        self.control.brake=abs(throttle_brake)
            else:
                # self.control.steer=np.clip(np.random.normal(self.control.steer,self.control_sigma['Steer']),-self.steer_bound,self.steer_bound)
                # if self.control.throttle>0:
                #     throttle_brake=self.control.throttle
                # else:
                #     throttle_brake=-self.control.brake
                # throttle_brake=np.clip(np.random.normal(throttle_brake,self.control_sigma['Throttle_brake']),-self.brake_bound,self.throttle_bound)
                # if throttle_brake>0:
                #     self.control.throttle=throttle_brake
                #     self.control.brake=0
                # else:
                #     self.control.throttle=0
                #     self.control.brake=abs(throttle_brake)
                pass

            # print(self.map.get_waypoint(self.ego_vehicle.get_location(),False),self.ego_vehicle.get_transform(),sep='\n')
            # print(self.sim_world.get_snapshot().timestamp)
            # spectator = self.sim_world.get_spectator()
            # transform = self.ego_vehicle.get_transform()
            # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
            #                                         carla.Rotation(pitch=-90)))
            for _ in range(self.control.exec_steps):
                if self.is_effective_action():
                    con=carla.VehicleControl(throttle=self.control.throttle,steer=self.control.steer,brake=self.control.brake,hand_brake=False,reverse=self.control.reverse,
                            manual_gear_shift=self.control.manual_gear_shift,gear=self.control.gear)
                    self.ego_vehicle.apply_control(con)
                if self.pygame:
                    self._tick()
                else:
                    spectator = self.sim_world.get_spectator()
                    transform = self.ego_vehicle.get_transform()
                    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                            carla.Rotation(pitch=-90)))
                    self.sim_world.tick()

                #camera_data = self.sensor_queue.get(block=True)
            """Attention: the server's tick function only returns after it ran a fixed_delta_seconds, so the client need not to wait for
            the server, the world snapshot of tick returned already include the next state after the uploaded action."""
            # print(self.map.get_waypoint(self.ego_vehicle.get_location(),False),self.ego_vehicle.get_transform(),sep='\n')
            # print(self.sim_world.get_snapshot().timestamp)
            # print()
            # if self.is_effective_action():
            cont = self.ego_vehicle.get_control()
            print(f"Actual Control:{cont}, Exec_steps:{self.control.exec_steps}")
            # print(self.ego_vehicle.get_speed_limit(),get_speed(self.ego_vehicle,False),get_acceleration(self.ego_vehicle,False),sep='\t')
            # route planner
            self.next_wps, _, self.vehicle_front = self.local_planner.run_step()

            if self.debug:
                # run the ego vehicle with PID_controller
                if self.next_wps[0].id != self.former_wp.id:
                    self.former_wp = self.next_wps[0]

                draw_waypoints(self.sim_world, [self.next_wps[0]], 60, z=1)
                self.sim_world.debug.draw_point(self.ego_vehicle.get_location(), size=0.1, life_time=5.0)
                # control=self.controller.run_step({'waypoints':self.next_wps,'vehicle_front':self.vehicle_front})
                # print(control.steer,control.throttle,control.brake,sep='\t')
            else:
                #draw_waypoints(self.sim_world, self.next_wps, 1.0/self.fps+0.001, z=1)
                pass

            if self.ego_vehicle.get_location().distance(self.former_wp.transform.location) >= self.sampling_resolution:
                self.former_wp = self.next_wps[0]

            """Attention: The sequence of following code is pivotal, do not recklessly chage their execution order"""
            reward = self._get_reward()
            state = self._get_state({'waypoints': self.next_wps, 'vehicle_front': self.vehicle_front})
            self.step_info.update({'Reward': reward})
            self.last_acc = self.ego_vehicle.get_acceleration()
        else:
            temp = self.sim_world.wait_for_tick()
            self.sim_world.on_tick(lambda _: {})
            time.sleep(1.0 / self.fps)

        if self.debug:
            print(f"Speed:{get_speed(self.ego_vehicle, False)}, Acc:{get_acceleration(self.ego_vehicle, False)}")
        print(f"Current State:{self.speed_state}, RL In Control:{self.RL_switch}")
        if not self.RL_switch:
            print(f"Control Sigma -- Steer:{self.control_sigma['Steer']}, Throttle_brake:{self.control_sigma['Throttle_brake']}")
        if self.is_effective_action():
            # update timesteps
            self.time_step += 1
            self.total_step += 1
            if self.speed_state == SpeedState.RUNNING and self.RL_switch == True:
                self.rl_control_step += 1
            print(f"Ego Vehicle Speed Limit:{self.ego_vehicle.get_speed_limit() * 3.6}\n"
                  f"Episode:{self.reset_step}, Total_step:{self.total_step}, Time_step:{self.time_step}, RL_control_step:{self.rl_control_step}, \n"
                  f"Vel: {get_speed(self.ego_vehicle, False)}, Acc:{get_acceleration(self.ego_vehicle, False)}, distance:{state['vehicle_front'][0] * self.sampling_resolution * self.buffer_size}, \n"
                  f"Reward:{self.step_info['Reward']}, TTC:{self.step_info['TTC']}, Comfort:{self.step_info['Comfort']}, "
                  f"Efficiency:{self.step_info['Efficiency']}, Lane_center:{self.step_info['Lane_center']}, Yaw:{self.step_info['Yaw']} \n"
                  f"Steer:{self.control.steer}, Throttle:{self.control.throttle}, Brake:{self.control.brake}, Exec_steps:{self.control.exec_steps}")

            return state, reward, self._truncated(), self._done(), self._get_info({'Steer': self.control.steer, 'Throttle': self.control.throttle, 
                'Brake': self.control.brake, 'Exec_steps':self.control.exec_steps_info})
        else:
            return state, reward, self._truncated(), self._done(), self._get_info()

    def get_observation_space(self):
        """Get observation space of cureent environment
        first element: Next waypoints list length of ego vehicle,
        second element: Location (x,y) dimention of waypoint"""
        return {'waypoints': self.buffer_size, 'ego_vehicle': 6, 'companion_vehicle': 2}

    def get_action_bound(self):
        """Return action bound of ego vehicle controller"""
        return {'steer': self.steer_bound, 'throttle': self.throttle_bound, 'brake': self.brake_bound}

    def is_effective_action(self):
        # testing if current ego vehcle's action should be put into replay buffer
        return self.speed_state == SpeedState.REBOOT or self.speed_state == SpeedState.RUNNING

    def seed(self, seed=None):
        return

    def render(self, mode):
        pass

    def _get_state(self, dict):
        """return a tuple: the first element is next waypoints, the second element is vehicle_front information"""

        # The wps_length here is a litle tricky, compared with the commented version
        # wps_length=dict['waypoints'][-1].transform.location.distance(self.ego_vehicle.get_location())
        wps_length = self.sampling_resolution * self.buffer_size
        wps = []
        # print(self.ego_vehicle.get_transform().rotation,
        #     dict['waypoints'][0].road_id,dict['waypoints'][0].lane_id,dict['waypoints'][0].transform.rotation,
        #     dict['waypoints'][1].road_id,dict['waypoints'][1].lane_id,dict['waypoints'][1].transform.rotation,
        #     dict['waypoints'][2].road_id,dict['waypoints'][2].lane_id,dict['waypoints'][2].transform.rotation,sep='\t')
        if dict['waypoints']:
            for wp in dict['waypoints']:
                lane_center = get_lane_center(self.map,self.ego_vehicle.get_location())
                #distance = self.ego_vehicle.get_location().distance(wp.transform.location)
                distance = lane_center.transform.location.distance(wp.transform.location)
                yaw_diff = math.degrees(get_yaw_diff(wp.transform.get_forward_vector(),
                                                     self.ego_vehicle.get_transform().get_forward_vector()))
                yaw_diff/=90

                wps.append([distance / wps_length, yaw_diff])
        if len(wps) < self.buffer_size:
            # end of route, not enough next waypoints
            gap = self.buffer_size - len(wps)
            for _ in range(gap):
                wps.append([(len(wps) + 1) * self.sampling_resolution / wps_length, 0])

        if dict['vehicle_front']:
            vehicle_front = dict['vehicle_front']
            ego_speed = get_speed(self.ego_vehicle, False)
            vf_speed = get_speed(vehicle_front, False)
            rel_speed = ego_speed - vf_speed
            distance = self.ego_vehicle.get_location().distance(vehicle_front.get_location())
            vehicle_len=max(abs(self.ego_vehicle.bounding_box.extent.x),abs(self.ego_vehicle.bounding_box.extent.y))+ \
                max(abs(self.vehicle_front.bounding_box.extent.x),abs(self.vehicle_front.bounding_box.extent.y))
            distance -= vehicle_len
            if distance <self.min_distance:
                vfl=[0, rel_speed/5]
            else:
                distance -= self.min_distance
                vfl = [distance / (self.vehicle_proximity-self.min_distance), rel_speed/5]
        else:
            # No vehicle front, suppose there is a vehicle at the end of waypoint list and relative speed is 0
            vfl = [1, 0]

        # ego vehicle information
        lane_center = get_lane_center(self.map, self.ego_vehicle.get_location())
        right_lane_dis = lane_center.get_right_lane().transform.location.distance(self.ego_vehicle.get_location())
        t = lane_center.lane_width / 2 + lane_center.get_right_lane().lane_width / 2 - right_lane_dis

        yaw_diff_ego = math.degrees(get_yaw_diff(lane_center.transform.get_forward_vector(),
                                               self.ego_vehicle.get_transform().get_forward_vector()))

        yaw_forward = lane_center.transform.get_forward_vector()
        v_3d = self.ego_vehicle.get_velocity()
        theta_v=get_yaw_diff(v_3d,yaw_forward)
        v_s = v_3d.length() * math.cos(theta_v)
        v_t = v_3d.length() * math.sin(theta_v)
        # v_t1=v_3d.length()*math.cos(alpha_v)

        a_3d = self.ego_vehicle.get_acceleration()
        theta_a=get_yaw_diff(a_3d,yaw_forward)
        a_s = a_3d.length() * math.cos(theta_a)
        a_t = a_3d.length() * math.sin(theta_a)

        """Attention:
        Upon initializing, there are some bugs in the theta_v and theta_a, which could be greater than 90,
        this might be caused by carla."""
        return {'waypoints': wps, 'ego_vehicle': [v_s/10 , v_t/10 , a_s/3, a_t/3 , t, yaw_diff_ego/90],
                'vehicle_front': vfl}

    def _get_reward(self):
        """Calculate the step reward:
        TTC: Time to collide with front vehicle
        Eff: Ego vehicle efficiency, speed ralated
        Com: Ego vehicle comfort, ego vehicle acceration change rate
        Lcen: Distance between ego vehicle location and lane center
        """
        ego_speed = get_speed(self.ego_vehicle, True)
        lane_center = get_lane_center(self.map, self.ego_vehicle.get_location())
        TTC = self.TTC_THRESHOLD
        if self.vehicle_front:
            distance = self.ego_vehicle.get_location().distance(self.vehicle_front.get_location())
            vehicle_len=max(abs(self.ego_vehicle.bounding_box.extent.x),abs(self.ego_vehicle.bounding_box.extent.y))+ \
                max(abs(self.vehicle_front.bounding_box.extent.x),abs(self.vehicle_front.bounding_box.extent.y))
            distance -= vehicle_len
            if distance < self.min_distance:
                TTC = 0.000001
            else:
                distance -= self.min_distance
                rel_speed = ego_speed / 3.6 - get_speed(self.vehicle_front, False)
                if abs(rel_speed)> float(0.0000001):
                    TTC = distance / rel_speed
            #print(distance, TTC)
        # fTTC=-math.exp(-TTC)
        if TTC >= 0 and TTC <= self.TTC_THRESHOLD:
            fTTC = np.clip(np.log(TTC / self.TTC_THRESHOLD),-1,0)
        else:
            fTTC = 0

        yaw_forward = lane_center.transform.get_forward_vector().make_unit_vector()
        v_3d = self.ego_vehicle.get_velocity()
        # ignore z value
        v_3d.z = 0
        if v_3d.length() != 0.0:
            theta_v = math.acos(np.clip(v_3d.dot(yaw_forward) / (v_3d.length() * yaw_forward.length()), -1, 1))
            # alpha_v = math.acos(np.clip(v_3d.dot(yaw_right)/(v_3d.length()*yaw_right.length()),-1,1))
        else:
            theta_v = math.acos(0)
            # alpha_v=math.acos(0)
        v_s = v_3d.length() * math.cos(theta_v)
        if v_s*3.6 > self.speed_limit:
            # fEff = 1
            fEff = math.exp(self.speed_limit - v_s * 3.6)
        else:
            fEff = v_s * 3.6 / self.speed_limit

        cur_acc = self.ego_vehicle.get_acceleration()
        jerk = (cur_acc.x - self.last_acc.x) ** 2 / (1.0 / self.fps) + (cur_acc.y - self.last_acc.y) ** 2 / (
                1.0 / self.fps)
        # The max jerk here is 1200, whick still requires further testing
        fCom = -jerk / (36 * self.fps ** 2)

        Lcen = lane_center.transform.location.distance(self.ego_vehicle.get_location())
        print(
            f"Lane Center:{Lcen}, Road ID:{lane_center.road_id}, Lane ID:{lane_center.lane_id}, Yaw:{self.ego_vehicle.get_transform().rotation.yaw}")

        if not test_waypoint(lane_center) or Lcen > lane_center.lane_width / 2:
            fLcen = -1.5
        else:
            fLcen = - Lcen / (lane_center.lane_width / 2)

        yaw_diff=math.degrees(get_yaw_diff(lane_center.transform.get_forward_vector(),
                                self.ego_vehicle.get_transform().get_forward_vector()))
        fYaw= -abs(yaw_diff)/90

        self.step_info = {'velocity':v_s,'offlane':Lcen, 'yaw_diff':yaw_diff,'TTC': TTC, 'fTTC':fTTC, 'Comfort': fCom, 'Efficiency': fEff, 'Lane_center': fLcen, 'Yaw': fYaw, 'Abandon':False}

        if self._truncated():
            history, tags = self.collision_sensor.get_collision_history()
            if len(history) != 0:
                if SemanticTags.Car in tags or SemanticTags.Truck in tags or SemanticTags.Bus in tags or SemanticTags.Motorcycle in tags \
                        or SemanticTags.Rider in tags or SemanticTags.Bicycle in tags:
                    return - self.penalty
                else:
                    # If ego vehicle collides with traffic lights and stop signs, do not add penalty
                    self.step_info['Abandon']=True
                    return (fTTC + fEff)*2 + fLcen +fYaw
            else:
                return - self.penalty
        else:
            return (fTTC+fEff)*2 + fCom  + fLcen +fYaw

    def _speed_switch(self):
        """cont: the control command of RL agent"""
        ego_speed = get_speed(self.ego_vehicle)
        if self.speed_state == SpeedState.START:
            # control=self.controller.run_step({'waypoints':self.next_wps,'vehicle_front':self.vehicle_front})
            if ego_speed >= self.speed_threshold:
                self.speed_state = SpeedState.RUNNING
                self._ego_autopilot(False)
                if not self.RL_switch:
                    #Under basic agent control
                    self.autopilot_controller.set_destination(random.choice(self.spawn_points).location)
                    control=self.autopilot_controller.run_step()
        elif self.speed_state == SpeedState.RUNNING:
            if self.RL_switch == True:
                # if ego_speed < self.speed_min and self.vehicle_front:
                #     distance = self.ego_vehicle.get_location().distance(self.vehicle_front.get_location())
                #     vehicle_len=max(abs(self.ego_vehicle.bounding_box.extent.x),abs(self.ego_vehicle.bounding_box.extent.y))+ \
                #         max(abs(self.vehicle_front.bounding_box.extent.x),abs(self.vehicle_front.bounding_box.extent.y))
                #     distance -= vehicle_len
                #     if distance<self.min_distance+1:
                #         #Ego vehicle following front vehicle
                #         self.speed_state = SpeedState.REBOOT
                pass
            else:
                #Under basic agent control
                if self.autopilot_controller.done() and self.loop:
                    self.autopilot_controller.set_destination(random.choice(self.spawn_points).location)
                control=self.autopilot_controller.run_step()
        elif self.speed_state == SpeedState.REBOOT:
            #control = self.controller.run_step({'waypoints': self.next_wps, 'vehicle_front': self.vehicle_front})
            if ego_speed >= self.speed_threshold:
                # self._ego_autopilot(False)
                self.speed_state = SpeedState.RUNNING
        else:
            logging.error('CODE LOGIC ERROR')

        if self.speed_state==SpeedState.RUNNING:
            if not self.RL_switch:
                self.control.brake=control.brake
                self.control.throttle=control.throttle
                self.control.steer=control.steer
                self.control.exec_steps=1
                self.control.exec_steps_info=-1.0
        else:
            self.control.exec_steps=1
            self.control.exec_steps_info=-1.0

    def _truncated(self):
        """Calculate whether to terminate the current episode"""
        if len(self.collision_sensor.get_collision_history()[0]) != 0:
            # Here we judge speed state because there might be collision event when spawning vehicles
            logging.warn('collison happend')
            return True
        if self.map.get_waypoint(self.ego_vehicle.get_location()) is None:
            logging.warn('vehicle drive out of road')
            return True
        if get_speed(self.ego_vehicle, True) < self.speed_min and self.speed_state == SpeedState.RUNNING and \
                self.vehicle_front is None:
            logging.warn('vehicle speed too low')
            return True
        # if self.lane_invasion_sensor.get_invasion_count()!=0:
        #     logging.warn('lane invasion occur')
        #     return True
        if self.step_info['Lane_center'] == -1.5:
            logging.warn('lane invasion occur')
            return True

        return False

    def _done(self):
        if self.RL_switch == True and self.next_wps[2].transform.location.distance(
                self.ego_spawn_point.location) < self.sampling_resolution:
            # The local planner's waypoint list has been depleted
            logging.info('vehicle reach destination, simulation terminate')
            return True
        if not self.RL_switch:
            if self.time_step > 5000:
                # Let the traffic manager only execute 5000 steps. or it can fill the replay buffer
                logging.info('5000 steps passed under traffic manager control')
                return True
            if self.next_wps[2].transform.location.distance(
                    self.ego_spawn_point.location) < self.sampling_resolution:
                # The second next waypoints is close enough to the spawn point, route done
                logging.info('vehicle reach destination, simulation terminate')
                return True

        return False

    def _get_info(self, control_info=None):
        """Rerurn simulation running information,
            param: control_info, the current controller information
        """
        if control_info is None:
            return self.step_info
        else:
            self.step_info.update(control_info)
            return self.step_info

    def _ego_autopilot(self, setting=True):
        # Use traffic manager to control ego vehicle
        self.ego_vehicle.set_autopilot(setting, self.tm_port)
        if setting:
            speed_diff = (30 * 3.6 - self.speed_limit) / (30 * 3.6) * 100
            self.traffic_manager.distance_to_leading_vehicle(self.ego_vehicle, self.min_distance)
            self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, 100)
            self.traffic_manager.ignore_signs_percentage(self.ego_vehicle, 100)
            self.traffic_manager.ignore_vehicles_percentage(self.ego_vehicle, 0)
            self.traffic_manager.ignore_walkers_percentage(self.ego_vehicle, 100)
            self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, speed_diff)

            # self.traffic_manager.set_desired_speed(self.ego_vehicle,36)
            # ego_wp=self.map.get_waypoint(self.ego_vehicle.get_location())
            # self.traffic_manager.set_path(self.ego_vehicle,path)
            """set_route(self, actor, path):
                Sets a list of route instructions for a vehicle to follow while controlled by the Traffic Manager. 
                The possible route instructions are 'Left', 'Right', 'Straight'.
                The traffic manager only need this instruction when faces with a junction."""
            self.traffic_manager.set_route(self.ego_vehicle,
                                           ['Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight'])

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

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn a  vehicle at specific transform
        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            ego_bp = create_vehicle_blueprint(self.sim_world,self.ego_filter,ego=True,color='0,255,0')
            vehicle = self.sim_world.try_spawn_actor(ego_bp, transform)
            if vehicle is None:
                logging.warn("Ego vehicle generation fail")

        # if self.debug and vehicle:
        #      vehicle.show_debug_telemetry()

        return vehicle

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

            # print(transform)
            blueprint = create_vehicle_blueprint(self.sim_world, 'vehicle.audi.etron', ego=False, color='0,0,0', number_of_wheels=[4])
            # Spawn the cars and their autopilot all together
            command_batch.append(SpawnActor(blueprint, transform).
                                 then(SetAutopilot(FutureActor, True, self.tm_port)))

        # execute the command batch
        for (i, response) in enumerate(self.client.apply_batch_sync(command_batch, self.sync)):
            if response.has_error():
                logging.warn(response.error)
            else:
                # print("Future Actor",response.actor_id)
                self.companion_vehicles.append(self.sim_world.get_actor(response.actor_id))
                self.traffic_manager.ignore_lights_percentage(
                    self.sim_world.get_actor(response.actor_id), 50)
                self.traffic_manager.ignore_signs_percentage(
                    self.sim_world.get_actor(response.actor_id), 50)
                self.traffic_manager.ignore_walkers_percentage(
                    self.sim_world.get_actor(response.actor_id), 100)
                self.traffic_manager.set_route(self.sim_world.get_actor(response.actor_id),
                                               ['Straight', 'Straight', 'Straight', 'Straight', 'Straight'])
                self.traffic_manager.update_vehicle_lights(
                    self.sim_world.get_actor(response.actor_id),True)
                self.traffic_manager.vehicle_percentage_speed_difference(
                    self.sim_world.get_actor(response.actor_id), -0)
                # print(self.sim_world.get_actor(response.actor_id).attributes)

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
            if self.camera is not None:
                self.camera.stop()
            if self.collision_sensor is not None:
                self.collision_sensor.sensor.stop()
            if self.lane_invasion_sensor is not None:
                self.lane_invasion_sensor.sensor.stop()
            for actor_filter in actor_filters:
                self.client.apply_batch([carla.command.DestroyActor(x)
                                         for x in self.sim_world.get_actors().filter(actor_filter)])

        # for actor_filter in actor_filters:
        #     for actor in self.sim_world.get_actors().filter(actor_filter):
        #         if actor.is_alive:
        #             if actor.type_id =='controller.ai.walker':
        #                 actor.stop()
        #             actor.destroy()
