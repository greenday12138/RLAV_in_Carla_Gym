import logging
import torch
import datetime, os, sys
import random, collections
import numpy as np
import matplotlib.pyplot as plt
curPath=os.path.abspath(os.path.dirname(__file__))
rootPath=os.path.split(os.path.split(curPath)[0])[0]
sys.path.append(rootPath)
from tqdm import tqdm
from algs.ddpg import DDPG
from gym_carla.single_lane.settings import ARGS
from gym_carla.single_lane.carla_env import CarlaEnv
from main.util.process import start_process, kill_process

# neural network hyper parameters
SIGMA = 0.5
THETA = 0.05
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.9  # q值更新系数
TAU = 0.01  # 软更新参数
EPSILON = 0.5  # epsilon-greedy
BUFFER_SIZE = 40000
MINIMAL_SIZE = 128
BATCH_SIZE = 128
REPLACE_A = 500
REPLACE_C = 300
TOTAL_EPISODE = 50000
SIGMA_DECAY = 0.9999
TTC_threshold = 4.001
base_name = f'origin_{TTC_threshold}_NOCA'
time=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
SAVE_PATH=f"./out/single_lane/ddpg/{time}"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def main():
    args = ARGS.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # env=gym.make('CarlaEnv-v0')
    env = CarlaEnv(args)

    done = False
    truncated = False

    random.seed(0)
    torch.manual_seed(16)
    s_dim = env.get_observation_space()
    a_bound = env.get_action_bound()
    if args.adapt:
        a_dim = 3
    else:
        a_dim = 2 

    n_run = 3
    rosiolling_window = 100  # 100 car following events, average score
    result = []

    for run in [base_name]:
        agent = DDPG(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                     LR_CRITIC, DEVICE)

        # training part
        max_rolling_score = np.float('-5')
        max_score = np.float('-5')
        var = 3
        collision_train = 0
        episode_score = []
        rolling_score = []
        cum_collision_num = []

        score_safe = []
        score_efficiency = []
        score_comfort = []

        try:
            for i in range(10):
                with tqdm(total=TOTAL_EPISODE // 10, desc="Iteration %d" % i) as pbar:
                    for i_episode in range(TOTAL_EPISODE // 10):
                        state = env.reset()
                        agent.reset_noise()
                        score = 0
                        score_s, score_e, score_c = 0, 0, 0  # part objective scores

                        while not done and not truncated:
                            action = agent.take_action(state)

                            next_state, reward, truncated, done, info = env.step(action)
                            if env.is_effective_action() and not info['Abandon']:
                                if 'Throttle' in info:
                                    # Input the guided action to replay buffer
                                    throttle_brake = -info['Brake'] if info['Brake'] > 0 else info['Throttle']
                                    if args.adapt:
                                        action = np.array([[info['Steer'], throttle_brake, info['Exec_steps']]])
                                    else:
                                        action = np.array([[info['Steer'], throttle_brake]])
                                    agent.store_transition(state,action,reward,next_state,truncated,done,info)
                                else:
                                    # Input the agent action to repla y buffer
                                    agent.store_transition(state,action,reward,next_state,truncated,done,info)
                                    
                                print(f"state -- vehicle_front:{state['vehicle_front']}\n"
                                      f"waypoints:{state['waypoints']}, \n"
                                      f"ego_vehicle:{state['ego_vehicle']}, \n"
                                      f"next_state -- vehicle_front:{next_state['vehicle_front']}\n"
                                      f"waypoints:{next_state['waypoints']}\n"
                                      f"ego_vehicle:{next_state['ego_vehicle']}\n"
                                      f"action:{action}\n"
                                      f"reward:{reward}\n"
                                      f"truncated:{truncated}, done:{done}")
                                print()

                            if agent.replay_buffer.size() > MINIMAL_SIZE:
                                logging.info("Learn begin %f" % SIGMA)
                                agent.learn()

                            state = next_state
                            score += reward
                            score_s += info['fTTC']
                            score_e += info['Efficiency']
                            score_c += info['Comfort']

                            if env.total_step==args.pre_train_steps:
                                agent.save_net(f"{SAVE_PATH}/ddpg_pre_trained.pth")

                            if env.rl_control_step > 10000 and env.is_effective_action() and \
                                    env.RL_switch and SIGMA > 0.1:
                                globals()['SIGMA'] *= SIGMA_DECAY
                                agent.set_sigma(SIGMA)

                        if done or truncated:
                            # restart the training
                            done = False
                            truncated = False

                        # record episode results
                        episode_score.append(score)
                        score_safe.append(score_s)
                        score_efficiency.append(score_e)
                        score_comfort.append(score_c)
                        # rolling_score.append(np.mean(episode_score[max]))
                        cum_collision_num.append(collision_train)

                        if max_score < score:
                            max_score = score

                        """ if rolling_score[rolling_score.__len__-1]>max_rolling_score:
                            max_rolling_score=rolling_score[rolling_score.__len__-1]
                            agent.save_net() """

                        # result.append([episode_score,rolling_score,cum_collision_num,score_safe,score_efficiency,score_comfort])
                        if (i_episode + 1) % 10 == 0:
                            pbar.set_postfix({
                                'episodes': '%d' % (TOTAL_EPISODE / 10 * i + i_episode + 1),
                                'score': '%.2f' % score
                            })
                        pbar.update(1)

                        # sigma decay
                        # if agent.replay_buffer.size()>MINIMAL_SIZE:
                        #     globals()['SIGMA']*=SIGMA_DECAY
                        #     agent.set_sigma(SIGMA)

                    # sigma decay
                    # if agent.replay_buffer.size()>MINIMAL_SIZE:
                    #     globals()['SIGMA']*=SIGMA_DECAY
                    #     agent.set_sigma(SIGMA)

            agent.save_net(f"{SAVE_PATH}/ddpg_final.pth")
            np.save(f"{SAVE_PATH}/result_{run}.npy", result)
        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        # except BaseException as e:
        #      logging.info(e.args)
        finally:
            env.__del__()
            agent.save_net(f"{SAVE_PATH}/ddpg_final.pth")
            logging.info('\nDone.')


if __name__ == '__main__':
    try:
        start_process()
        main()
    # except BaseException as e:
    #     logging.warning(e.args)
    finally:
        kill_process()
