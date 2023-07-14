import logging
import torch
import datetime,time, os
import gym, macad_gym
import random, sys
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from copy import deepcopy
from collections import deque
from tensorboardX import SummaryWriter
from multiprocessing import Process, Queue, Pipe, connection, Lock
sys.path.append(os.getcwd())
from macad_gym.core.utils.wrapper import fill_action_param,recover_steer,Action
from algs.pdqn import P_DQN

# neural network hyper parameters
AGENT_PARAM = {
    "s_dim": {
        'waypoints': 10, 
        'ego_vehicle': 6, 
        'companion_vehicle': 3, 
        'light':3
    },
    "a_dim": 2,
    "a_bound":{
        'steer': 1.0, 
        'throttle': 1.0, 
        'brake': 1.0
    },
    "acc3": True,
    "Kaiming_normal": False,
    "buffer_size": 160000,
    "minimal_size": 10000,
    "batch_size": 256,
    "per_flag": True,
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    "sigma": 0.5,
    "sigma_steer": 0.3,
    "sigma_acc": 0.5,
    "sigma_decay": 0.9999,
    "theta": 0.05,
    "lr_actor": 0.0002,
    "lr_critic": 0.0002,
    "gamma": 0.9,   # q值更新系数
    "tau": 0.01,    # 软更新参数
    "epsilon": 0.5, # epsilon-greedy
    "clip_grad": 10,
    "zero_index_gradients": True,
    "inverting_gradients": True,
}
TRAIN = True
UPDATE_FREQ = 100
modify_change_steer=False
base_name = f'origin_NOCA'
time=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
SAVE_PATH=f"./out/multi_agent/pdqn/{time}"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    # env=gym.make('CarlaEnv-v0')
    env = gym.make("HomoNcomIndePoHiwaySAFR2CTWN5-v0")

    done = False
    truncated = False

    random.seed(0)
    torch.manual_seed(16)

    episode_writer=SummaryWriter(SAVE_PATH)
    n_run = 3
    rosiolling_window = 100  # 100 car following events, average score
    result = []

    for run in [base_name]:
        param = deepcopy(AGENT_PARAM)
        worker_agent = P_DQN(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                            param["tau"], param["sigma_steer"], param["sigma"], param["sigma_acc"], 
                            param["theta"], param["epsilon"], param["buffer_size"], param["batch_size"], 
                            param["lr_actor"], param["lr_critic"], param["clip_grad"], param["zero_index_gradients"],
                            param["inverting_gradients"], param["per_flag"], param["device"])
        # learner_agent = P_DQN(deepcopy(s_dim), a_dim, a_bound, GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
        #              LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients,PER_FLAG, DEVICE)

        #multi-process training
        process=list()
        traj_q=Queue(maxsize=BUFFER_SIZE)
        agent_q=Queue(maxsize=1)
        traj_send,traj_recv=Pipe()
        agent_send,agent_recv=Pipe()
        #process.append(mp.Process(target=learner_mp,args=(traj_recv,agent_send,(deepcopy(s_dim), a_dim, a_bound),Args.ego_num)))
        process.append(mp.Process(target=learner_mp,args=(traj_q,agent_q,(deepcopy(s_dim), a_dim, a_bound),Args.ego_num)))
        [p.start() for p in process]

        # training part
        max_rolling_score = np.float32('-5')
        max_score = np.float32('-30')
        var = 3
        collision_train = 0
        learn_time=0
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
                        while not done and not truncated:
                            actions,action_params,all_action_params=[],[],[]
                            # if agent_recv.poll():
                            #     a,a_t,c,c_t=agent_recv.recv()
                            #     worker_agent.actor.load_state_dict(a)
                            #     worker_agent.actor_target.load_state_dict(a_t)
                            #     worker_agent.critic.load_state_dict(c)
                            #     worker_agent.critic_target.load_state_dict(c_t)
                            if not agent_q.empty():
                                actor,actor_t,critic,critic_t=worker_agent.actor.state_dict(),worker_agent.actor_target.state_dict(),\
                                    worker_agent.critic.state_dict(),worker_agent.critic_target.state_dict()
                                # temp_agent,learn_time=agent_q.get()
                                # a,c=temp_agent.actor.state_dict(),temp_agent.critic.state_dict()
                                # worker_agent.actor.load_state_dict(temp_agent.actor.state_dict())
                                # worker_agent.critic.load_state_dict(temp_agent.critic.state_dict())
                                actor,critic,learn_time=agent_q.get()
                                worker_agent.actor=actor
                                worker_agent.critic=critic

                            for state in states:
                                action, action_param, all_action_param = worker_agent.take_action(state)
                                actions.append(action)
                                action_params.append(action_param)
                                all_action_params.append(all_action_param)
                            next_states, rewards, truncateds, dones, infos = env.step(actions, action_params)
                            for j in range(len(next_states)):
                                if env.is_effective_action(j) and not infos[j]['Abandon']:
                                    logging.info(f"CLIENT {j} INFO, LEARN TIME:{learn_time}")
                                    # traj_send.send((j,states[j],next_states[j],all_action_params[j],
                                    #      rewards[j],truncateds[j],dones[j],infos[j]))
                                    #if not traj_q.full():
                                    traj_q.put((deepcopy(j),deepcopy(states[j]),deepcopy(next_states[j]),deepcopy(all_action_params[j]),
                                        deepcopy(rewards[j]),deepcopy(truncateds[j]),deepcopy(dones[j]),deepcopy(infos[j])),block=True,timeout=None)
   
                                    print(
                                        f"state -- vehicle_info:{states[j]['vehicle_info']}\n"
                                        #f"waypoints:{states[j]['left_waypoints']}, \n"
                                        #f"waypoints:{states[j]['center_waypoints']}, \n"
                                        #f"waypoints:{states[j]['right_waypoints']}, \n"
                                        f"ego_vehicle:{states[j]['ego_vehicle']}, \n"
                                        f"light info: {states[j]['light']}\n"
                                        f"next_state -- vehicle_info:{next_states[j]['vehicle_info']}\n"
                                        #f"waypoints:{next_states[j]['left_waypoints']}, \n"
                                        #f"waypoints:{next_states[j]['center_waypoints']}, \n"
                                        #f"waypoints:{next_states[j]['right_waypoints']}, \n"
                                        f"ego_vehicle:{next_states[j]['ego_vehicle']}\n"
                                        f"light info: {next_states[j]['light']}\n"
                                        f"action:{actions[j]}, action_param:{action_params[j]}, all_action_param:{all_action_params[j]}\n"
                                        f"reward:{rewards[j]}, truncated:{truncateds[j]}, done:{dones[j]}")
                                    print()

                            for t in truncateds:
                                if t:
                                    truncated=True
                            for d in dones:
                                if d and not truncated:
                                    done=True
                            states = next_states
                            
                            #only record the first vehicle reward
                            if env.ego_clients[0].total_step == Args.pre_train_steps:
                                worker_agent.save_net(f"{SAVE_PATH}/pdqn_pre_trained.pth")
                            if env.is_effective_action(0) and not infos[0]['Abandon']:
                                score += rewards[0]
                                if not truncateds[0]:
                                    ttc += infos[0]['fTTC']
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
                                worker_agent.save_net(F"{SAVE_PATH}/pdqn_optimal.pth")

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
                        worker_agent.save_net(f"{SAVE_PATH}/pdqn_final.pth")
           
            np.save(f"{SAVE_PATH}/result_{run}.npy", result)
        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        # except BaseException as e:
        #      logging.info(e.args)
        finally:
            env.close()
            #process[-1].join() # waiting for learner
            episode_writer.close()
            worker_agent.save_net(f"{SAVE_PATH}/pdqn_final.pth")
            #process[-1].join()
            process_safely_terminate(process)
            logging.info('\nDone.')

def process_safely_terminate(process: list):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            logging.SystemError(e)

#Pipe version multiprocess
# def learner_mp(traj_recv:connection.Connection, agent_send:connection.Connection, agent_param, ego_num):
#     learner_agent=P_DQN(agent_param[0], agent_param[1], agent_param[2], GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
#                      LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients,PER_FLAG, DEVICE)
#     impact_deques=[deque(maxlen=2) for _ in range(ego_num)]
#     while(True):
#         if traj_recv.poll(timeout=None):
#             trajectory=traj_recv.recv()
#             ego_id, state, next_state, all_action_param, reward, truncated, done, info=trajectory[0],trajectory[1],trajectory[2],trajectory[3],\
#                 trajectory[4],trajectory[5],trajectory[6],trajectory[7]
#             replay_buffer_adder(learner_agent,impact_deques[ego_id],state,next_state,all_action_param,reward,truncated,done,info)

#         if learner_agent.replay_buffer.size()>=MINIMAL_SIZE:
#             logging.info("LEARN BEGIN")
#             learner_agent.learn()
#             if learner_agent.learn_time!=0 and learner_agent.learn_time%2==0:
#                 actor,actor_t,critic,critic_t=learner_agent.actor.state_dict(),learner_agent.actor_target.state_dict(), \
#                     learner_agent.critic.state_dict(),learner_agent.critic_target.state_dict()
#                 a,a_t,c,c_t=deepcopy(learner_agent.actor.state_dict()),deepcopy(learner_agent.actor_target.state_dict()),\
#                     deepcopy(learner_agent.critic.state_dict()),deepcopy(learner_agent.critic_target.state_dict())
#                 agent_send.send((a,a_t,c,c_t))

#Queue vesion multiprocess
def learner_mp(lock:Lock, traj_q: Queue, agent_q:Queue, agent_param, ego_num):
    param = deepcopy(AGENT_PARAM)
    learner = P_DQN(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                        param["tau"], param["sigma_steer"], param["sigma"], param["sigma_acc"], 
                        param["theta"], param["epsilon"], param["buffer_size"], param["batch_size"], 
                        param["lr_actor"], param["lr_critic"], param["clip_grad"], param["zero_index_gradients"],
                        param["inverting_gradients"], param["per_flag"], param["device"])
    if TRAIN and os.path.exists(f"./model_params/{SAVE_PATH}_net_params.pth"):
        learner.load_state_dict(torch.load(f"./model_params/{SAVE_PATH}_net_params.pth", map_location=param["device"]))
    update_count=0

    while(True):
        #alter the batch_size and update times according to the replay buffer size:
        #reference: https://zhuanlan.zhihu.com/p/345353294, https://arxiv.org/abs/1711.00489
        k = max(learner.replay_buffer.size()// param["minimal_size"], 1)
        learner.batch_size = k * param["batch_size"]
        for _ in range(UPDATE_FREQ):
            trajectory=traj_q.get(block=True,timeout=None)
            state, next_state, all_action_param, reward, done, truncated, info = trajectory[0], \
                trajectory[1], trajectory[2], trajectory[3], trajectory[4], trajectory[5], trajectory[6]
            
            replay_buffer_adder(learner_agent,impact_deques[ego_id],state,next_state,all_action_param,reward,truncated,done,info)        
        if learner.replay_buffer.size()>=param["minimal_size"]:
            logging.info("LEARN BEGIN")
            #print(f"LEARN TIME:{learner_agent.learn_time}")
            [learner_agent.learn() for _ in range(k)]
            if update_count!=0 and update_count//UPDATE_FREQ>0:
                update_count%=UPDATE_FREQ
                if not agent_q.full():
                    actor=deepcopy(learner_agent.actor).to('cpu')
                    critic=deepcopy(learner_agent.critic).to('cpu')
                    temp_agent.actor.load_state_dict(learner_agent.actor.state_dict())
                    temp_agent.critic.load_state_dict(learner_agent.critic.state_dict())
                    # actor,actor_t,critic,critic_t=learner_agent.actor.state_dict(),learner_agent.actor_target.state_dict(), \
                    #     learner_agent.critic.state_dict(),learner_agent.critic_target.state_dict()
                    # a,a_t,c,c_t=temp_agent.actor.state_dict(),temp_agent.actor_target.state_dict(), \
                    #     temp_agent.critic.state_dict(),temp_agent.critic_target.state_dict()
                    agent_q.put((actor,critic,learner_agent.learn_time),block=True,timeout=None)
                    #agent_q.put((temp_agent,learner_agent.learn_time),block=True,timeout=None)
        update_count+=1

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
        mp.set_start_method(method='spawn',force=True)  # force all the multiprocessing to 'spawn' methods
        main()
    except BaseException as e:
        logging.warning(e.args)