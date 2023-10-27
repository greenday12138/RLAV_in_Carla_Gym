import random
import carla
from macad_gym.viz.logger import LOG
from macad_gym.core.simulator.carla_provider import CarlaConnector, CarlaError


# TODO make the seed user configurable
random.seed(10)

def hero_autopilot(actor, traffic_manager, actor_config, env_config,setting=True):
    # Use traffic manager to control hero vehicle
    actor.set_autopilot(setting, traffic_manager.get_port())
    if setting:
        traffic_manager.distance_to_leading_vehicle(actor, env_config["min_distance"])
        if env_config['ignore_traffic_light']:
            traffic_manager.ignore_lights_percentage(actor, 100)
            traffic_manager.ignore_walkers_percentage(actor, 100)
        traffic_manager.ignore_signs_percentage(actor, 100)
        traffic_manager.ignore_vehicles_percentage(actor, 0)
        traffic_manager.vehicle_percentage_speed_difference(actor, 
                                                (30 - actor_config['speed_limit']) / 30 * 100)
        #traffic_manager.auto_lane_change(actor, True)
        traffic_manager.random_left_lanechange_percentage(actor, 100)
        traffic_manager.random_right_lanechange_percentage(actor, 100)

        # traffic_manager.set_desired_speed(actor, 72)
        # ego_wp=map.get_waypoint(actor.get_location())
        # traffic_manager.set_path(actor,path)
        """set_route(self, actor, path):
            Sets a list of route instructions for a vehicle to follow while controlled by the Traffic Manager. 
            The possible route instructions are 'Left', 'Right', 'Straight'.
            The traffic manager only need this instruction when faces with a junction."""
        traffic_manager.set_route(actor,
            ['Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight'])


def apply_traffic(world, traffic_manager, env_config, num_vehicles, num_pedestrians, safe=False, route_points=None):
    # set traffic manager
    #traffic_manager.set_synchronous_mode(env_config["sync_server"])
    if env_config["hybrid"] is True:
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(500)
        traffic_manager.set_respawn_dormant_vehicles(True)
        #To enable respawning of dormant vehicles within 25 and 700 meters of the hero vehicle
        traffic_manager.set_boundaries_respawn_dormant_vehicles(200, 2000)

    # --------------
    # Spawn vehicles
    # --------------

    blueprints = world.get_blueprint_library().filter("vehicle.*")
    if safe:
        blueprints = list(filter(lambda x: int(x.get_attribute('number_of_wheels')) == 4 and not
                (#x.id.endswith('microlino') or
                 x.id.endswith('carlacola') or
                 x.id.endswith('cybertruck') or
                 x.id.endswith('t2') or
                 x.id.endswith('t2_2021') or
                 x.id.endswith('fusorosa') or
                 #x.id.endswith('sprinter') or
                 x.id.endswith('firetruck') or
                 x.id.endswith('ambulance')), blueprints))

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    if route_points is None:
        spawn_points = world.get_map().get_spawn_points()
    else:
        spawn_points = route_points
    number_of_spawn_points = len(spawn_points)

    random.shuffle(spawn_points)
    if num_vehicles <= number_of_spawn_points:
        spawn_points = random.sample(spawn_points, num_vehicles)
    else:
        msg = ''
        LOG.traffic_logger.warning(f"requested {num_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points")
        num_vehicles = number_of_spawn_points

    vehicles_list = []
    failed_v = 0
    for n, transform in enumerate(spawn_points):
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            #color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', '0,0,0')
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # spawn npc vehicles
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            vehicles_list.append(vehicle)
        else:
            failed_v += 1

    LOG.traffic_logger.info("{}/{} vehicles correctly spawned.".format(num_vehicles-failed_v, num_vehicles))

    # Set automatic vehicle lights update if specified
    # if args.car_lights_on:
    #     all_vehicle_actors = world.get_actors(vehicles_id_list)
    #     for actor in all_vehicle_actors:
    #         traffic_manager.update_vehicle_lights(actor, True)

    # -------------
    # Spawn Walkers
    # -------------
    percentagePedestriansRunning = 0.0  # how many pedestrians will run
    percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
    blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
    pedestrian_controller_bp = world.get_blueprint_library().find('controller.ai.walker')

    # Take all the random locations to spawn
    spawn_points = []
    for i in range(num_pedestrians):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # Spawn the walker object
    pedestrians_list = []
    controllers_list = []
    pedestrians_speed = []
    failed_p = 0
    for spawn_point in spawn_points:
        pedestrian_bp = random.choice(blueprints)
        # set as not invincible
        if pedestrian_bp.has_attribute('is_invincible'):
            pedestrian_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if pedestrian_bp.has_attribute('speed'):
            if random.random() > percentagePedestriansRunning:
                speed = pedestrian_bp.get_attribute('speed').recommended_values[1]  # walking
            else:
                speed = pedestrian_bp.get_attribute('speed').recommended_values[2]  # running
        else:
            speed = 0.0
        pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
        if pedestrian is not None:
            controller = world.try_spawn_actor(pedestrian_controller_bp, carla.Transform(), pedestrian)
            if controller is not None:
                pedestrians_list.append(pedestrian)
                controllers_list.append(controller)
                pedestrians_speed.append(speed)
            else:
                pedestrian.destroy()
                failed_p += 1
        else:
            failed_p += 1

    LOG.traffic_logger.info("{}/{} pedestrians correctly spawned.".format(num_pedestrians-failed_p, num_pedestrians))
    
    # -------------
    # Set Traffic Lights
    # -------------
    lights_list=world.get_actors().filter("*traffic_light*")
    for light in lights_list:
        light.set_green_time(30)
        light.set_red_time(0)
        light.set_yellow_time(0)
    
    world.tick()
    #CarlaConnector.tick(world, LOG.traffic_logger)

    # Initialize each controller and set target to walk
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i, controller in enumerate(controllers_list):
        controller.start()  # start walker
        controller.go_to_location(world.get_random_location_from_navigation())  # set walk to random point
        controller.set_max_speed(float(pedestrians_speed[int(i / 2)]))  # max speed
    
    # Initialize each npc vehicle
    for veh in vehicles_list:
        traffic_manager.distance_to_leading_vehicle(veh, env_config["min_distance"])
        traffic_manager.set_route(veh,
                        ['Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight'])
        traffic_manager.update_vehicle_lights(veh, True)
        traffic_manager.ignore_signs_percentage(veh, 100)
        traffic_manager.ignore_lights_percentage(veh, 100 if env_config["ignore_traffic_light"] else 0)
        traffic_manager.auto_lane_change(veh, env_config["auto_lane_change"])
        # modify change probability
        traffic_manager.random_left_lanechange_percentage(veh, 0)
        traffic_manager.random_right_lanechange_percentage(veh, 0)
        traffic_manager.vehicle_percentage_speed_difference(veh,
                random.choice([-100,-100,-100,-140,-160,-180]))

    return vehicles_list, (pedestrians_list, controllers_list)
