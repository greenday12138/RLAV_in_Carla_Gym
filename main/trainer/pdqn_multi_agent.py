import logging
import torch
import datetime
import random, collections
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from collections import deque
from algs.pdqn import P_DQN
from tensorboardX import SummaryWriter
from multiprocessing import Process,Queue,Pipe
from gym_carla.multi_agent.settings import ARGS
from gym_carla.multi_agent.carla_env import CarlaEnv
from main.util.process import start_process, kill_process
from gym_carla.multi_agent.util.wrapper import fill_action_param,recover_steer,Action

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
MINIMAL_SIZE = 40000
BATCH_SIZE = 128
REPLACE_A = 500
REPLACE_C = 300
TOTAL_EPISODE = 50000
SIGMA_DECAY = 0.9999
TTC_threshold = 4.001
PER_FLAG=True
modify_change_steer=False
clip_grad = 10
zero_index_gradients = True
inverting_gradients = True
base_name = f'origin_{TTC_threshold}_NOCA'
SAVE_PATH='./out'

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

    time=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    episode_writer=SummaryWriter(f"{SAVE_PATH}/multi_agent/runs/pdqn/{time}")
    n_run = 3
    rosiolling_window = 100  # 100 car following events, average score
    result = []

    for run in [base_name]:
        worker_agent = P_DQN(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                     LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients,PER_FLAG, DEVICE)
        learner_agent = P_DQN(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                     LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients,PER_FLAG, DEVICE)

        #multi-process training
        process=list()
        traj_pipe=Queue(maxsize=MINIMAL_SIZE)
        agent_pipe=Queue(maxsize=BATCH_SIZE)
        mp.set_start_method(method='spawn',force=True)  # force all the multiprocessing to 'spawn' methods
        process.append(mp.Process(target=learner_mp,args=(learner_agent,traj_pipe,agent_pipe)))
        [p.start() for p in process]

        # training part
        max_rolling_score = np.float32('-5')
        max_score = np.float32('-30')
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
                        worker_agent.reset_noise()
                        score = 0
                        ttc, efficiency,comfort,lcen,yaw,impact,lane_change_reward = 0, 0, 0, 0, 0, 0, 0  # part objective scores
                        impact_deques=[deque(maxlen=2) for _ in range(len(states))]
                        while not done and not truncated:
                            actions,action_params,all_action_params=[],[],[]
                            if not agent_pipe.empty():
                                actor,actor_t,critic,critic_t=agent_pipe.get()
                                print(actor,actor_t,critic,critic_t,sep='\n')
                                worker_agent.actor.load_state_dict()
                                worker_agent.actor_target.load_state_dict()
                                worker_agent.critic.load_state_dict()
                                worker_agent.critic_target.load_state_dict()

                            for state in states:
                                action, action_param, all_action_param = worker_agent.take_action(state)
                                actions.append(action)
                                action_params.append(action_param)
                                all_action_params.append(all_action_param)
                            next_states, rewards, truncateds, dones, infos = env.step(actions, action_params)
                            for i in range(len(next_states)):
                                if env.is_effective_action(i) and not infos[i]['Abandon']:
                                    logging.info(f"CLIENT {i} INFO")
                                    if not traj_pipe.full():
                                        traj_pipe.put((worker_agent,impact_deques[i],states[i],next_states[i],all_action_params[i],
                                            rewards[i],truncateds[i],dones[i],infos[i]))
   
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
                                    print()

                            for t in truncateds:
                                if t:
                                    truncated=True
                            for d in dones:
                                if d and not truncated:
                                    done=True
                            states = next_states
                            
                            #only record the first vehicle reward
                            if env.ego_clients[0].total_step == args.pre_train_steps:
                                worker_agent.save_net(f"{SAVE_PATH}/multi_agent/pdqn_pre_trained.pth")
                            if env.is_effective_action(0) and not infos[0]['Abandon']:
                                score += rewards[0]
                                if not truncateds[0]:
                                    ttc += infos[0]['TTC']
                                    efficiency += infos[0]['Efficiency']
                                    comfort += infos[0]['Comfort']
                                    lcen += infos[0]['Lane_center']
                                    yaw += infos[0]['Yaw']
                                    impact += infos[0]['impact']
                                    lane_change_reward += infos[0]['lane_changing_reward']

                            if env.is_effective_action():
                                print(f"Joint RL control steps:{env.rl_control_step}")     
                                if env.rl_control_step > 10000 and SIGMA_ACC > 0.01:
                                    globals()['SIGMA'] *= SIGMA_DECAY
                                    globals()['SIGMA_STEER'] *= SIGMA_DECAY
                                    globals()['SIGMA_ACC'] *= SIGMA_DECAY
                                    worker_agent.set_sigma(SIGMA_STEER, SIGMA_ACC)
                                    logging.info("Agent Sigma %f %f", SIGMA_STEER,SIGMA_ACC)
                           
                        if done or truncated:
                            # restart the training
                            done = False
                            truncated = False

                        # record episode results
                        if env.ego_clients[0].RL_switch:
                            episode_writer.add_scalar('Total_Reward',score,i*(TOTAL_EPISODE // 10)+i_episode)
                            time_step=env.ego_clients[0].time_step
                            score/=time_step+1
                            episode_writer.add_scalar('Avg_Reward',score,i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Time_Steps',time_step,i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('TTC',ttc/(time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Efficiency',efficiency/(time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Comfort',comfort/(time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Lcen',lcen/(time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Yaw',yaw/(time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Impact',impact/(time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            episode_writer.add_scalar('Lane_change_reward',lane_change_reward/(time_step+1), i*(TOTAL_EPISODE // 10)+i_episode)
                            
                            episode_score.append(score)
                            score_safe.append(ttc)
                            score_efficiency.append(efficiency)
                            score_comfort.append(comfort)
                            # rolling_score.append(np.mean(episode_score[max]))
                            cum_collision_num.append(collision_train)

                            if max_score < score:
                                max_score = score
                                worker_agent.save_net(F"{SAVE_PATH}/multi_agent/pdqn_optimal.pth")

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
                        worker_agent.save_net(f"{SAVE_PATH}/multi_agent/pdqn_final.pth")
           
            np.save(f"{SAVE_PATH}/multi_agent/result_{run}.npy", result)
        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        # except BaseException as e:
        #      logging.info(e.args)
        finally:
            env.__del__()
            #process[-1].join() # waiting for learner
            episode_writer.close()
            worker_agent.save_net(f"{SAVE_PATH}/multi_agent/pdqn_final.pth")
            process_safely_terminate(process)
            logging.info('\nDone.')

def process_safely_terminate(process: list):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            logging.SystemError(e)

def learner_mp(agent, traj_q: Queue, agent_q:Queue):
    while(True):        
        if agent.replay_buffer.size()>=MINIMAL_SIZE:
            if not traj_q.empty():
                trajectory=traj_q.get(block=True,timeout=1.0)
                print(trajectory)
                impact_deque, state, next_state, all_action_param, reward, truncated, done, info=trajectory[0],trajectory[1],trajectory[2],trajectory[3],\
                    trajectory[4],trajectory[5],trajectory[6],trajectory[7]
                replay_buffer_adder(agent,impact_deque,state,next_state,all_action_param,reward,truncated,done,info)
            logging.info("LEARN BEGIN")
            agent.learn()
            if agent.learn_time!=0 and agent.learn_time%BATCH_SIZE==0:
                agent_q.put((agent.actor.state_dict(),agent.actor_target.state_dict(),
                            agent.critic.state_dict(),agent.critic_target.state_dict()),block=True,timeout=None)
        else:
            trajectory=traj_q.get(block=True,timeout=None)
            impact_deque, state, next_state, all_action_param, reward, truncated, done, info=trajectory[0],trajectory[1],trajectory[2],trajectory[3],\
            trajectory[4],trajectory[5],trajectory[6],trajectory[7]
            replay_buffer_adder(agent,impact_deque,state,next_state,all_action_param,reward,truncated,done,info)

def replay_buffer_adder(agent,impact_deque, state, next_state,all_action_param,reward, truncated, done, info):
    """Input all the state info into agent's replay buffer"""
    if 'Throttle' in info:
        control_state = info['control_state']
        throttle_brake = -info['Brake'] if info['Brake'] > 0 else info['Throttle']
        if info['Change']==Action.LANE_FOLLOW:
            action=1
        elif info['Change']==Action.LANE_CHANGE_LEFT:
            action=0
        elif info['Change']==Action.LANE_CHANGE_RIGHT:
            action=2
        # action_param = np.array([[info['Steer'], throttle_brake]])
        saved_action_param = fill_action_param(action, info['Steer'], throttle_brake,
                                                all_action_param,modify_change_steer)
        print(f"Control In Replay Buffer: {action}, {saved_action_param}")
        if control_state:
            # under rl control
            if truncated:
                agent.store_transition(state, action, saved_action_param, reward, next_state,
                                    truncated, done, info)
            else:
                impact = info['impact'] / 9
                impact_deque.append([state, action, saved_action_param, reward, next_state,
                                        truncated, done, info])
                if len(impact_deque) == 2:
                    experience = impact_deque[0]
                    agent.store_transition(experience[0], experience[1], experience[2],
                                            experience[3] + impact, experience[4], experience[5],
                                            experience[6], experience[7])
                # agent.replay_buffer.add(state, action, saved_action_param, reward, next_state,
                #                         truncated, done, info)
        else:
            # Input the guided action to replay buffer
            if truncated:
                agent.store_transition(state,action,saved_action_param,reward,next_state,
                    truncated,done,info)
            else:
                impact = info['impact'] / 9
                impact_deque.append([state, action, saved_action_param, reward, next_state,
                                        truncated, done, info])
                if len(impact_deque) == 2:
                    experience = impact_deque[0]
                    agent.store_transition(experience[0], experience[1], experience[2],
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
