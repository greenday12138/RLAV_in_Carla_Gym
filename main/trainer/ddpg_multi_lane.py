import logging
import torch
import datetime, os
import random, collections
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from algs.ddpg_multi_lane import DDPG
from gym_carla.multi_lane.settings import ARGS
from tensorboardX import SummaryWriter
from gym_carla.multi_lane.carla_env import CarlaEnv
from main.util.process import start_process, kill_process

# neural network hyper parameters
SIGMA = 0.5
THETA = 0.05
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LR_ACTOR = 0.0001
LR_CRITIC = 0.0002
GAMMA = 0.9  # q值更新系数
TAU = 0.01  # 软更新参数
EPSILON = 0.5  # epsilon-greedy
BUFFER_SIZE = 10000
MINIMAL_SIZE = 10000
BATCH_SIZE = 128
REPLACE_A = 500
REPLACE_C = 300
TOTAL_EPISODE = 5000
SIGMA_DECAY = 0.9999
TTC_threshold = 4.001
PER_FLAG=True
base_name = f'origin_{TTC_threshold}_NOCA'
time=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
SAVE_PATH=f"./out/multi_lane/ddpg/{time}"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def main():
    args = ARGS.parse_args()
    args.modify_change_steer=False
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
    a_dim = 2

    episode_writer=SummaryWriter(SAVE_PATH)
    n_run = 3
    rosiolling_window = 100  # 100 car following events, average score
    result = []

    for run in [base_name]:
        agent = DDPG(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE,
                LR_ACTOR,LR_CRITIC, PER_FLAG,DEVICE)

        # training part
        max_rolling_score = np.float32('-5')
        max_score = np.float32('-5')
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
                        ttc, efficiency,comfort,lcen,yaw,impact,lane_change_reward = 0, 0, 0, 0, 0, 0, 0  # part objective scores

                        while not done and not truncated:
                            action = agent.take_action(state)
                            next_state, reward, truncated, done, info = env.step(0, action)
                            if env.is_effective_action() and not info['Abandon']:
                                if 'Throttle' in info:
                                    # Input the guided action to replay buffer
                                    throttle_brake = -info['Brake'] if info['Brake'] > 0 else info['Throttle']
                                    action = np.array([[info['Steer'], throttle_brake]])
                                    agent.store_transition(state,action,reward,next_state,truncated,done,info)
                                else:
                                    # not work
                                    # Input the agent action to replay buffer
                                    agent.store_transition(state,action,reward,next_state,truncated,done,info)
                                   
                                print(
                                        f"state -- vehicle_info:{state['vehicle_info']}\n"
                                        #f"waypoints:{state['left_waypoints']}, \n"
                                        f"waypoints:{state['center_waypoints']}, \n"
                                        #f"waypoints:{state['right_waypoints']}, \n"
                                        f"ego_vehicle:{state['ego_vehicle']}, \n"
                                        f"light info: {state['light']}\n"
                                        f"next_state -- vehicle_info:{next_state['vehicle_info']}\n"
                                        #f"waypoints:{next_state['left_waypoints']}, \n"
                                        f"waypoints:{next_state['center_waypoints']}, \n"
                                        #f"waypoints:{next_state['right_waypoints']}, \n"
                                        f"ego_vehicle:{next_state['ego_vehicle']}\n"
                                        f"light info: {next_state['light']}\n"
                                        f"action:{action}, reward:{reward}, truncated:{truncated}, done:{done}")
                            print()

                            if agent.replay_buffer.size() >= MINIMAL_SIZE:
                                logging.info("Learn begin %f" % SIGMA)
                                agent.learn()

                            state = next_state
                            score += reward
                            if not truncated:
                                ttc += info['TTC']
                                efficiency += info['Efficiency']
                                comfort += info['Comfort']
                                lcen += info['Lane_center']
                                yaw += info['Yaw']
                                impact += info['impact']
                                lane_change_reward += info['lane_changing_reward']

                            if env.total_step==args.pre_train_steps:
                                agent.save_net(f"{SAVE_PATH}/pdqn_pre_trained.pth")

                            if env.rl_control_step > 10000 and env.is_effective_action() and \
                                    env.RL_switch and SIGMA > 0.01:
                                globals()['SIGMA'] *= SIGMA_DECAY
                                agent.set_sigma(SIGMA)

                        if done or truncated:
                            # restart the training
                            done = False
                            truncated = False

                        # record episode results
                        if env.RL_switch:
                            episode_writer.add_scalar('Total_Reward',score,i*(TOTAL_EPISODE // 10)+i_episode)
                            score/=env.time_step+1
                            episode_writer.add_scalar('Avg_Reward',score,i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Time_Steps',env.time_step,i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('TTC',ttc/(env.time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Efficiency',efficiency/(env.time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Comfort',comfort/(env.time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Lcen',lcen/(env.time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Yaw',yaw/(env.time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Impact',impact/(env.time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Lane_change_reward',lane_change_reward/(env.time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            
                            episode_score.append(score)
                            score_safe.append(ttc)
                            score_efficiency.append(efficiency)
                            score_comfort.append(comfort)
                            # rolling_score.append(np.mean(episode_score[max]))
                            cum_collision_num.append(collision_train)

                            if max_score < score:
                                max_score = score
                                agent.save_net(F"{SAVE_PATH}/pdqn_optimal.pth")

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
                        agent.save_net(f"{SAVE_PATH}/pdqn_final.pth")

            np.save(f"{SAVE_PATH}/result_{run}.npy", result)
        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        # except BaseException as e:
        #      logging.info(e.args)
        finally:
            env.__del__()
            episode_writer.close()
            agent.save_net(f"{SAVE_PATH}/pdqn_final.pth")
            logging.info('\nDone.')


if __name__ == '__main__':
    try:
        start_process()
        main()
    # except BaseException as e:
    #     logging.warning(e.args)
    finally:
        kill_process()
