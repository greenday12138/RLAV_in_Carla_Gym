"""This file defines all high level parameters of carla gym environment"""
import argparse

# the following road id sets define the chosen route
ROADS = set()
STRAIGHT = {8, 0, 1, 2, 3, 15, 5, 6, 7}
CURVE = {11, 13, 20, 14}
JUNCTION = {40, 41, 61, 62, 117, 118, 93, 94, 157, 158}
# JUNCTION_LANE={33,85,141}
# JUNCTION_LANE_MINUS={102,109,150,163,46,67,128}
ROADS.update(STRAIGHT)
ROADS.update(CURVE)
ROADS.update(JUNCTION)

# the flowing arguments set the simulation parameters
ARGS = argparse.ArgumentParser(
    description='CARLA_gym Client')
ARGS.add_argument(
    '-v', '--verbose',
    action='store_true',
    dest='debug',
    default=False,
    help='Print debug information')
ARGS.add_argument(
    '-t',
    action='store_true',
    dest='train',
    default=True,
    help='Training Reinforcement agent')
ARGS.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
ARGS.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
ARGS.add_argument(
    '--res',
    metavar='WIDTHxHEIGHT',
    default='1280x720',
    help='Window resolution (default: 1280x720)')
ARGS.add_argument(
    '--sync',
    action='store_true',
    default=True,
    help='Synchronous mode execution')
ARGS.add_argument(
    '--fps', metavar='FPS',
    default=10, type=int,
    help="The fps of server running speed")
ARGS.add_argument(
    '--filter',
    metavar='PATTERN',
    default='vehicle.tesla.model3',
    help='Actor filter (default: "vehicle.*")')
ARGS.add_argument(
    '-l', '--loop',
    action='store_true',
    dest='loop',
    default='True',
    help='Sets a new random destination upon reaching the previous one (default: False)')
ARGS.add_argument(
    "-a", "--agent", type=str,
    choices=["Behavior", "Basic"],
    help="select which agent to run",
    default="Behavior")
ARGS.add_argument(
    '-b', '--behavior', type=str,
    choices=["cautious", "normal", "aggressive"],
    help='Choose one of the possible agent behaviors (default: normal) ',
    default='normal')
ARGS.add_argument(
    '-s', '--seed',
    help='Set seed for repeating executions (default: None)',
    default=None,
    type=int)
ARGS.add_argument(
    '-m', '--map', type=str,
    choices=['Town01_Opt', 'Town02_Opt'],
    help='Choose one of the possible world maps',
    default='Town01_Opt')
ARGS.add_argument(
    '-n', '--num_of_vehicles', type=list,
    help='Total vehicles number which run in simulation',
    default=[5,10,15,20])
ARGS.add_argument(
    '-sa', '--sampling_resolution', type=float,
    help='Distance between generated two waypoints',
    default=1.0)
ARGS.add_argument(
    '--tm-port',
    metavar='P',
    default=8000,
    type=int,
    help='Port to communicate with traffic manager (default: 8000)')
ARGS.add_argument(
    '--hybrid',
    action='store_true',
    default=True,
    help='Activate hybrid mode for Traffic Manager')
ARGS.add_argument(
    '--no_rendering',
    action='store_true',
    default=False,
    help='Activate no rendering mode')
ARGS.add_argument(
    '--stride', type=int,
    default=10,
    help='The number of upfront waypoints each state should include')
ARGS.add_argument(
    '--buffer-size', type=int,
    default=30,
    help='The number of look-ahead waypoints in each step')
ARGS.add_argument(
    '--TTC_th', type=float,
    default=4.001,
    help='TTC threshold')
ARGS.add_argument(
    '--penalty', type=float,
    default=20,
    help='reward penalty for simulation terminated early on account of collision and lane invasion')
ARGS.add_argument(
    '--speed_limit', type=float,
    default=72.0,
    help='Speed limit for ego vehicle, km/h')
ARGS.add_argument(
    '--speed_threshold', type=float,
    default=10.0,
    help='Speed limit for ego vehicle, km/h')
ARGS.add_argument(
    '--speed_min', type=float,
    default=0.01,
    help='When ego vehicle speed reaches down to this threshold, we should let basic agent take control \
        and the action of basic need to add into the replay buffer, km/h')
ARGS.add_argument(
    '--steer_bound', type=float,
    default=1.0,
    help='Steer bound for ego vehicle controller')
ARGS.add_argument(
    '--throttle_bound', type=float,
    default=1.0,
    help='Throttle bound for ego vehicle controller')
ARGS.add_argument(
    '--brake_bound', type=float,
    default=1.0,
    help='Brake bound for ego vehicle controller')
ARGS.add_argument(
    '--switch_threshold', type=int,
    default=20,
    help='Let the RL controller and PID controller alternatively take control every 500 steps'
)
ARGS.add_argument(
    '--pre_train_steps', type=int,
    default=10000,
    help='Let the RL controller and PID controller alternatively take control every 500 steps'
)
ARGS.add_argument(
    '--vehicle_proximity',type=float,
    default=30.0,
    help='Distance for searching vehicles in front of ego vehicle, unit -- meters'
)
ARGS.add_argument(
    '--min_distance',type=float,
    default=2.0,
    help='Min distance between two vehicles, unit -- meters'
)