import logging
import torch
import random, collections
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from algs.pdqn import P_DQN
from gym_carla.multi_lane.settings import ARGS
from gym_carla.multi_lane.carla_env import CarlaEnv, SpeedState
from main.util.process import start_process, kill_process

# neural network hyper parameters
SIGMA = 1
SIGMA_STEER = 0.3
SIGMA_ACC = 0.5
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
TTC_threshold = 4.001
clip_grad = 10
zero_index_gradients = True
inverting_gradients = True
base_name = f'origin_{TTC_threshold}_NOCA'


def main():
    args = ARGS.parse_args()
    args.pre_train_stpes=0
    args.no_rendering = False

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
    a_dim = 2

    result = []

    for run in [base_name]:
        param = torch.load('./out/pdqn_final.pth')
        agent = P_DQN(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                     LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients, DEVICE)
        agent.load_net(param)
        agent.train = False
        env.RL_switch=True
        agent.set_sigma(0,0)

        state = env.reset()

        try:
            while not done and not truncated:
                action,action_param,all_action_param = agent.take_action(state)
                next_state, reward, truncated,done, info = env.step(action,action_param)
                # if env.speed_state == SpeedState.REBOOT:
                #     env.speed_state = SpeedState.RUNNING
                state=next_state
                print()

        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        finally:
            env.__del__()
            logging.info('\nDone.')


if __name__ == '__main__':
    try:
        start_process()
        main()
    except BaseException as e:
        logging.warning(e.args)
    finally:
        kill_process()
