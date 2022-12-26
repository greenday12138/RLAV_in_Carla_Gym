import logging
import torch
import random, collections
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from algs.pdqn import P_DQN
from gym_carla.multi_agent.settings import ARGS
from gym_carla.multi_agent.carla_env import CarlaEnv
from main.util.process import start_process, kill_process
from gym_carla.multi_agent.util.wrapper import fill_action_param,recover_steer
from collections import deque

# neural network hyper parameters
SIGMA = 0.5
SIGMA_STEER = 0.3
SIGMA_ACC = 0.5
THETA = 0.05
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LR_ACTOR = 0.0002
LR_CRITIC = 0.0002
GAMMA = 0.9  # q值更新系数
TAU = 0.01  # 软更新参数
EPSILON = 0.5  # epsilon-greedy
BUFFER_SIZE = 40000
MINIMAL_SIZE = 10000
BATCH_SIZE = 128
REPLACE_A = 500
REPLACE_C = 300
TOTAL_EPISODE = 50000
SIGMA_DECAY = 0.9999
TTC_threshold = 4.001
modify_change_steer=False
clip_grad = 10
zero_index_gradients = True
inverting_gradients = True
base_name = f'origin_{TTC_threshold}_NOCA'


def main():
    args = ARGS.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # env=gym.make('CarlaEnv-v0')
    env = CarlaEnv(args)
    globals()['modify_change_steer'] = args.modify_change_steer

    done = False
    truncated = False

    random.seed(0)
    torch.manual_seed(16)
    s_dim = env.get_observation_space()
    a_bound = env.get_action_bound()
    a_dim = 2

    n_run = 3
    rosiolling_window = 100  # 100 car following events, average score
    result = []

    for run in [base_name]:
        agent = P_DQN(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                     LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients, DEVICE)

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
                        states = env.reset()
                        agent.reset_noise()
                        score = 0
                        score_s, score_e, score_c = 0, 0, 0  # part objective scores
                        impact_deques=[deque(maxlen=2) for _ in range(len(states))]
                        while not done and not truncated:
                            actions,action_params,all_action_params=[],[],[]
                            for state in states:
                                action, action_param, all_action_param = agent.take_action(state)
                                actions.append(action)
                                action_params.append(action_param)
                                all_action_params.append(all_action_param)
                            next_states, rewards, truncateds, dones, infos = env.step(actions, action_params)
                            for i in range(len(next_states)):
                                if env.is_effective_action(i) and not infos[i]['Abandon']:
                                    logging.info(f"CLIENT {i} INFO")
                                    replay_buffer_adder(agent,impact_deques[i],states[i],next_states[i],actions[i],all_action_params[i],
                                        rewards[i],truncateds[i],dones[i],infos[i])
   
                                    print(
                                        f"state -- vehicle_info:{states[i]['vehicle_info']}\n"
                                        #f"waypoints:{state['left_waypoints']}, \n"
                                        #f"waypoints:{states[i]['center_waypoints']}, \n"
                                        #f"waypoints:{state['right_waypoints']}, \n"
                                        f"ego_vehicle:{states[i]['ego_vehicle']}, \n"
                                        f"light info: {states[i]['light']}\n"
                                        f"next_state -- vehicle_info:{next_states[i]['vehicle_info']}\n"
                                        #f"waypoints:{next_state['left_waypoints']}, \n"
                                        #f"waypoints:{next_states[i]['center_waypoints']}, \n"
                                        #f"waypoints:{next_state['right_waypoints']}, \n"
                                        f"ego_vehicle:{next_states[i]['ego_vehicle']}\n"
                                        f"light info: {next_states[i]['light']}\n"
                                        f"action:{actions[i]}, action_param:{action_params[i]}, all_action_param:{all_action_params[i]}\n"
                                        f"reward:{rewards[i]}, truncated:{truncateds[i]}, done:{dones[i]}")
                            
                            # if env.total_step == args.pre_train_steps:
                            #     agent.save_net('./out/pdqn_pre_trained.pth') 
                            if env.is_effective_action():
                                print(f"Joint RL control steps:{env.rl_control_step}")     
                                if env.rl_control_step > 10000 and SIGMA_ACC > 0.01:
                                    globals()['SIGMA'] *= SIGMA_DECAY
                                    globals()['SIGMA_STEER'] *= SIGMA_DECAY
                                    globals()['SIGMA_ACC'] *= SIGMA_DECAY
                                    agent.set_sigma(SIGMA_STEER, SIGMA_ACC)
                                
                            print()

                            if agent.replay_buffer.size() > MINIMAL_SIZE:
                                logging.info("Learn begin: %f %f", SIGMA_STEER,SIGMA_ACC)
                                agent.learn()

                            for t in truncateds:
                                if t:
                                    truncated=True
                            for d in dones:
                                if d and not truncated:
                                    done=True
                            states = next_states
                            # score += reward
                            
                            # if not truncated:
                            #     score_s += info['TTC']
                            #     score_e += info['Efficiency']
                            #     score_c += info['Comfort']
                        if done or truncated:
                            # restart the training
                            done = False
                            truncated = False

                        # record episode results
                        # episode_score.append(score)
                        # score_safe.append(score_s)
                        # score_efficiency.append(score_e)
                        # score_comfort.append(score_c)
                        # rolling_score.append(np.mean(episode_score[max]))
                        # cum_collision_num.append(collision_train)

                        # if max_score < score:
                        #     max_score = score

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

            agent.save_net('./out/pdqn_final.pth')
            np.save(f'./out/result_{run}.npy', result)
        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        # except BaseException as e:
        #      logging.info(e.args)
        finally:
            env.__del__()
            logging.info('\nDone.')

def replay_buffer_adder(agent,impact_deque, state, next_state, action,all_action_param,reward, truncated, done, info):
    """Input all the state info into agent's replay buffer"""
    if 'Throttle' in info:
        control_state = info['control_state']
        if control_state:
            # under rl control
            if truncated:
                agent.replay_buffer.add(state, action, all_action_param, reward, next_state,
                                    truncated, done, info)
            else:
                impact = info['impact'] / 9
                impact_deque.append([state, action, all_action_param, reward, next_state,
                                        truncated, done, info])
                if len(impact_deque) == 2:
                    experience = impact_deque[0]
                    agent.replay_buffer.add(experience[0], experience[1], experience[2],
                                            experience[3] + impact, experience[4], experience[5],
                                            experience[6], experience[7])
                # agent.replay_buffer.add(state, action, all_action_param, reward, next_state,
                #                         truncated, done, info)
                print('rl control in replay buffer: ', action, all_action_param)
        else:
            # Input the guided action to replay buffer
            throttle_brake = -info['Brake'] if info['Brake'] > 0 else info['Throttle']
            action = info['Change']
            # action_param = np.array([[info['Steer'], throttle_brake]])
            saved_action_param = fill_action_param(action, info['Steer'], throttle_brake,
                                                    all_action_param,modify_change_steer)
            print('agent control in replay buffer: ', action, saved_action_param)
            if truncated:
                agent.replay_buffer.add(state, action, saved_action_param, reward, next_state,
                                    truncated, done, info)
            else:
                impact = info['impact'] / 9
                impact_deque.append([state, action, saved_action_param, reward, next_state,
                                        truncated, done, info])
                if len(impact_deque) == 2:
                    experience = impact_deque[0]
                    agent.replay_buffer.add(experience[0], experience[1], experience[2],
                                            experience[3] + impact, experience[4], experience[5],
                                            experience[6], experience[7])
                # agent.replay_buffer.add(state, action, saved_action_param, reward, next_state,
                #                         truncated, done, info)
    # else:
    #     # not work
    #     # Input the agent action to replay buffer
    #     agent.replay_buffer.add(state, action, all_action_param, reward, next_state, truncated, done, info)

if __name__ == '__main__':
    try:
        start_process()
        main()
    # except BaseException as e:
    #     logging.warning(e.args)
    finally:
        kill_process()
