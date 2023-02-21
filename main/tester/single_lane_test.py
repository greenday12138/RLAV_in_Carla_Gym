import logging
import torch
import datetime, os
import random, collections
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from algs.ddpg import DDPG
from algs.td3 import TD3
from tensorboardX import SummaryWriter
from gym_carla.single_lane.settings import ARGS
from gym_carla.single_lane.carla_env import CarlaEnv, SpeedState
from main.util.process import start_process, kill_process

# neural network hyper parameters
SIGMA = 1
THETA = 0.001
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.99  # q值更新系数
TAU = 0.01  # 软更新参数
EPSILON = 0.5  # epsilon-greedy
POLICY_UPDATE_FREQ = 5
BUFFER_SIZE = 20000
MINIMAL_SIZE = 10000
BATCH_SIZE = 128
REPLACE_A = 500
REPLACE_C = 300
TOTAL_EPISODE = 3000
base_name = f'origin_NOCA'
time=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
SAVE_PATH=f"./out/single_lane/ddpg/test/{time}"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def main():
    ARGS.set_defaults(train=False)
    ARGS.set_defaults(no_rendering=False)
    args = ARGS.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # env=gym.make('CarlaEnv-v0')
    env = CarlaEnv(args)

    done = False
    truncated=False

    random.seed(0)
    torch.manual_seed(8)
    s_dim = env.get_observation_space()
    a_bound = env.get_action_bound()
    if args.adapt:
        a_dim = 3
    else:
        a_dim = 2 

    result = []
    episode_writer=SummaryWriter(SAVE_PATH)

    for run in [base_name]:
        # param = torch.load('./out/td3_pre_trained.pth')
        # agent = TD3(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
        #     LR_CRITIC, POLICY_UPDATE_FREQ, DEVICE)
        param = torch.load('./out/ddpg_pre_trained.pth')
        agent = DDPG(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                     LR_CRITIC, DEVICE)
        agent.load_net(param)
        agent.set_sigma(0)

        VEL=[]
        JERK=[]
        OFFLANE=[]
        TTC=[]
        

        try:
            for i in range(30):
                state = env.reset()

                while not done and not truncated:
                    action = agent.take_action(state)
                    next_state, reward, truncated,done, info = env.step(action)
                    # if env.speed_state == SpeedState.REBOOT:
                    #     env.speed_state = SpeedState.RUNNING
                    state=next_state
                    print()

                if done or truncated:
                    done = False
                    truncated = False

        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        except BaseException as e:
            logging.info(e.args)
        finally:
            env.__del__()
            episode_writer.close()
            logging.info('\nDone.')


if __name__ == '__main__':
    try:
        start_process()
        main()
    except BaseException as e:
        logging.warning(e.args)
    finally:
        kill_process()
