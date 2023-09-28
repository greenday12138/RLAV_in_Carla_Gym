import torch
import logging
import datetime,time, os
import gym, macad_gym
import random, sys
import traceback
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from copy import deepcopy
from collections import deque
from tensorboardX import SummaryWriter
from multiprocessing import Process, Queue, Lock
sys.path.append(os.getcwd())
from main.util.process import kill_process
from main.util.utils import get_gpu_info, get_gpu_mem_info
from macad_gym import LOG_PATH
from macad_gym.viz.logger import LOG
from macad_gym.core.simulator.carla_provider import CarlaError
from macad_gym.core.utils.wrapper import (fill_action_param, recover_steer, Action, 
    SpeedState, Truncated)
from algs.pdqn import P_DQN
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

# neural network hyper parameters
AGENT_PARAM = {
    "s_dim": {
        'waypoints': 10, 
        'hero_vehicle': 6, 
        'companion_vehicle': 4, 
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
TOTAL_EPISODE = 50000
TRAIN = True
UPDATE_FREQ = 100
modify_change_steer=False
base_name = f'origin_NOCA'
MODEL_PATH = os.path.join(os.getcwd(), 'out', 'model_params', 'pdqn_ma_net_params.pth')
SAVE_PATH = os.path.join(os.getcwd(), 'out', 'multi_agent', 'pdqn', 
                         datetime.datetime.today().strftime('%Y-%m-%d_%H-%M'))
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def main():
    env = gym.make("PDQNHomoNcomIndePoHiwaySAFR2CTWN5-v0")
    random.seed(0)
    torch.manual_seed(16)

    episode_writer=SummaryWriter(SAVE_PATH)
    result = []

    for run in [base_name]:
        param = deepcopy(AGENT_PARAM)
        worker = P_DQN(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                            param["tau"], param["sigma_steer"], param["sigma"], param["sigma_acc"], 
                            param["theta"], param["epsilon"], param["buffer_size"], param["batch_size"], 
                            param["lr_actor"], param["lr_critic"], param["clip_grad"], param["zero_index_gradients"],
                            param["inverting_gradients"], param["per_flag"], torch.device('cpu'))
        if TRAIN and os.path.exists(MODEL_PATH):
            worker.load_net(MODEL_PATH, map_location=worker.device)
            #worker.set_sigma(0.01, 0.01)

        #multi-process training
        process=list()
        lock=Lock()
        #traj_q=Queue(maxsize=10)
        traj_q=Queue(maxsize=param["minimal_size"])
        agent_q=Queue(maxsize=1)
        process.append(mp.Process(target=learner_mp,args=(lock, traj_q, agent_q, AGENT_PARAM)))
        [p.start() for p in process]

        # training part
        max_score = np.float32('-30')
        collision_train = 0
        learn_time=0
        episode_score = []
        cum_collision_num = []

        score_safe = []     
        score_efficiency = []
        score_comfort = []
        losses_episode = []
        total_reward, avg_reward = {}, {}
        ttc, efficiency, comfort, lcen, lane_change_reward = {}, {}, {}, {}, {}  # part objective scores

        try:
            for i in range(TOTAL_EPISODE//1000):
                with tqdm(total=1000, desc="Iteration %d" % i) as pbar:
                    for i_episode in range(1000):
                        try:
                            states, _ = env.reset()
                            # if i_episode > 2 and reload_agent(worker):
                            #     param = deepcopy(AGENT_PARAM)
                            #     worker = P_DQN(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                            #                     param["tau"], param["sigma_steer"], param["sigma"], param["sigma_acc"], 
                            #                     param["theta"], param["epsilon"], param["buffer_size"], param["batch_size"], 
                            #                     param["lr_actor"], param["lr_critic"], param["clip_grad"], param["zero_index_gradients"],
                            #                     param["inverting_gradients"], param["per_flag"], param["device"])
                            done, truncated = False, False
                            worker.reset_noise()
                            for actor_id in states.keys():
                                ttc[actor_id], efficiency[actor_id], comfort[actor_id], lcen[actor_id],\
                                    lane_change_reward[actor_id], total_reward[actor_id], avg_reward[actor_id] = \
                                    -1, -1, -1, -1, 0, 0, -4
                        
                            while not done and not truncated:
                                action_dict, actions, action_params, all_action_params={}, {}, {}, {}
                                if TRAIN and not agent_q.empty():
                                    lock.acquire()
                                    worker.load_net(os.path.join(SAVE_PATH, 'learner.pth'), map_location=worker.device)
                                    learn_time, q_loss = agent_q.get()
                                    lock.release()
                                    worker.learn_time=learn_time
                                    if q_loss is not None:
                                        LOG.rl_trainer_logger.info(f"PDQN LEARN TIME:{learn_time}, Q_loss:{q_loss}")
                                        losses_episode.append(q_loss)

                                for actor_id in states.keys():
                                    actions[actor_id], action_params[actor_id], all_action_params[actor_id
                                                            ] = worker.take_action(states[actor_id][1])
                                    action_dict[actor_id]={
                                        "action_index": actions[actor_id], "action_param": action_params[actor_id]}
                                next_states, rewards, dones, truncateds, infos = env.step(action_dict)
                                for actor_id in next_states.keys():
                                    if infos[actor_id]["speed_state"] == str(SpeedState.RUNNING):
                                        total_reward[actor_id] += infos[actor_id]["reward"]
                                        if truncateds[actor_id] == Truncated.FALSE:
                                            info = infos[actor_id]["reward_info"]
                                            ttc[actor_id] += info["ttc_reward"]
                                            efficiency[actor_id] += info["efficiency_reward"]
                                            comfort[actor_id] += info["comfort_reward"]
                                            lcen[actor_id] += info["lane_center_reward"]
                                            lane_change_reward[actor_id] += info["lane_change_reward"]

                                        #process action params
                                        state, next_state, reward, done, truncated, info = \
                                            deepcopy(states[actor_id][1]), deepcopy(next_states[actor_id][1]), \
                                            deepcopy(rewards[actor_id]), deepcopy(dones[actor_id]),\
                                            deepcopy(truncateds[actor_id]!=Truncated.FALSE), deepcopy(infos[actor_id])
                                            
                                        throttle_brake = -info["control_info"]["brake"] if info["control_info"]["brake"] > 0 else info["control_info"]["throttle"]
                                        if info['current_action']==str(Action.LANE_FOLLOW):
                                            action=1
                                        elif info['current_action']==str(Action.LANE_CHANGE_LEFT):
                                            action=0
                                        elif info['current_action']==str(Action.LANE_CHANGE_RIGHT):
                                            action=2
                                        saved_action_param = fill_action_param(action, info["control_info"]["steer"], throttle_brake,
                                                                            all_action_params[actor_id], modify_change_steer)
                                        LOG.rl_trainer_logger.debug(f"\nPDQN Control In Replay Buffer: {action}, {saved_action_param}")

                                        traj_q.put((state, next_state, action, saved_action_param,
                                            reward, done, truncated, info ), block=True, timeout=None)
                                        
                                        LOG.rl_trainer_logger.debug(
                                            f"PDQN\n"
                                            f"actor_id: {actor_id} time_steps: {info['step']}\n"
                                            f"state -- vehicle_info: {state['vehicle_info']}\n"
                                            #f"waypoints:{state['left_waypoints']}, \n"
                                            #f"waypoints:{state['center_waypoints']}, \n"
                                            #f"waypoints:{state['right_waypoints']}, \n"
                                            f"hero_vehicle: {state['hero_vehicle']}, \n"
                                            f"light info: {state['light']}\n"
                                            f"next_state -- vehicle_info:{next_state['vehicle_info']}\n"
                                            #f"waypoints:{next_state['left_waypoints']}, \n"
                                            #f"waypoints:{next_state['center_waypoints']}, \n"
                                            #f"waypoints:{next_state['right_waypoints']}, \n"
                                            f"hero_vehicle: {next_state['hero_vehicle']}\n"
                                            f"light info: {next_state['light']}\n"
                                            f"action: {actions[actor_id]}, action_param: {action_params[actor_id]} \n"
                                            f"all_action_param: {all_action_params[actor_id]},\n"
                                            f"saved_action_param: {saved_action_param}\n"
                                            f"reward: {reward}, truncated: {truncated}, done: {done}, ")
            
                                done = dones["__all__"]
                                truncated = truncateds["__all__"]!=Truncated.FALSE
                                states = next_states
                                
                                #only record the first vehicle reward
                                if env.unwrapped._total_steps == env.unwrapped.pre_train_steps:
                                    worker.save_net(os.path.join(SAVE_PATH, 'pdqn_pre_trained.pth'))
    
                                if env.unwrapped._rl_control_steps > 10000 and param["sigma_acc"] > 0.01:
                                    param["sigma"] *= param["sigma_decay"]
                                    param["sigma_steer"] *= param["sigma_decay"]
                                    param["sigma_acc"] *= param["sigma_decay"]
                                    worker.set_sigma(param["sigma_steer"], param["sigma_acc"])
                                    LOG.rl_trainer_logger.info(f"PDQN Agent Sigma {param['sigma_steer']} {param['sigma_acc']}")
                            
                            if done or truncated:
                                # restart the training
                                done = False
                                truncated = False

                            # record episode results
                            episodes = 1000 * i + i_episode
                            if env.unwrapped._rl_switch:
                                episode_writer.add_scalars("Total_Reward", total_reward, episodes)
                                for actor_id in total_reward.keys():
                                    avg_reward[actor_id] = total_reward[actor_id] / (env.unwrapped._time_steps[actor_id] + 1) 
                                    ttc[actor_id] /= env.unwrapped._time_steps[actor_id] + 1
                                    efficiency[actor_id] /= env.unwrapped._time_steps[actor_id] + 1
                                    comfort[actor_id] /= env.unwrapped._time_steps[actor_id] + 1
                                    lcen[actor_id] /= env.unwrapped._time_steps[actor_id] + 1
                                    lane_change_reward[actor_id] /= env.unwrapped._time_steps[actor_id] + 1
                                episode_writer.add_scalars('Avg_Reward', avg_reward, episodes)
                                episode_writer.add_scalars('Time_Steps', env.unwrapped._time_steps, episodes)
                                episode_writer.add_scalars('TTC', ttc, episodes)
                                episode_writer.add_scalars('Efficiency', efficiency, episodes)
                                episode_writer.add_scalars('Comfort', comfort, episodes)
                                episode_writer.add_scalars('Lcen', lcen, episodes)
                                episode_writer.add_scalars('Lane_change_reward', lane_change_reward, episodes)
                                
                                # score_safe.append(ttc)
                                # score_efficiency.append(efficiency)
                                # score_comfort.append(comfort)
                                # rolling_score.append(np.mean(episode_score[max]))
                                # cum_collision_num.append(collision_train)

                            LOG.rl_trainer_logger.info(f"PDQN Total_steps:{env.unwrapped._total_steps} RL_control_steps:{env.unwrapped._rl_control_steps}")

                            """ if rolling_score[rolling_score.__len__-1]>max_rolling_score:
                                max_rolling_score=rolling_score[rolling_score.__len__-1]
                                agent.save_net() """

                            pbar.set_postfix({
                                "episodes": f"{episodes + 1}"
                            })
                            #if (i_episode + 1) % 10 == 0:
                            # pbar.set_postfix({
                            #     'episodes': '%d' % (1000 * i + i_episode),
                            #     'score': '%.2f' % total_reward
                            # })
                            pbar.update(1)
                            worker.save_net(os.path.join(SAVE_PATH, 'pdqn_final.pth'))
                            if TRAIN and episodes % 1000 == 0:
                                worker.save_net(os.path.join(SAVE_PATH, f'pdqn_{episodes}_net_params.pth'))
                        except CarlaError as e:
                            LOG.rl_trainer_logger.exception("PDQN Carla Failed, restart carla!")
           
                # restart carla to clear garbage
                env.close()
        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        finally:
            env.close()
            [p.join() for p in process]
            episode_writer.close()
            worker.save_net(os.path.join(SAVE_PATH, 'pdqn_final.pth'))
            logging.info('\nDone.')

#Queue vesion multiprocess
def learner_mp(lock:Lock, traj_q: Queue, agent_q:Queue, agent_param:dict):
    param = deepcopy(agent_param)
    learner = P_DQN(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                        param["tau"], param["sigma_steer"], param["sigma"], param["sigma_acc"], 
                        param["theta"], param["epsilon"], param["buffer_size"], param["batch_size"], 
                        param["lr_actor"], param["lr_critic"], param["clip_grad"], param["zero_index_gradients"],
                        param["inverting_gradients"], param["per_flag"], param["device"])
    #load pre-trained model
    if TRAIN and os.path.exists(MODEL_PATH):
        learner.load_net(MODEL_PATH, map_location=learner.device)
    update_count = 0

    while(True):
        #alter the batch_size and update times according to the replay buffer size:
        #reference: https://zhuanlan.zhihu.com/p/345353294, https://arxiv.org/abs/1711.00489
        k = max(learner.replay_buffer.size()// param["minimal_size"], 1)
        learner.batch_size = k * param["batch_size"]
        if traj_q.qsize() >= k:
            for _ in range(k):
                trajectory=traj_q.get(block=True,timeout=None)
                state, next_state, action, saved_action_param, reward, done, truncated, info = \
                    trajectory[0], trajectory[1], trajectory[2], trajectory[3], trajectory[4], trajectory[5], \
                    trajectory[6], trajectory[7]

                learner.store_transition(state, action, saved_action_param, reward, next_state,
                                        truncated, done, info)       
        if TRAIN and learner.replay_buffer.size()>=param["minimal_size"]:
            q_loss = learner.learn()
            update_count += 1
            if not agent_q.full() and update_count//UPDATE_FREQ>0:
                lock.acquire()
                agent_q.put((deepcopy(learner.learn_time), deepcopy(q_loss)), block=True, timeout=None)
                learner.save_net(os.path.join(SAVE_PATH, 'learner.pth'))
                lock.release()
                update_count %= UPDATE_FREQ

def reload_agent(agent, gpu_id=0):
    if agent.device != torch.device('cpu'):
        return False
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=gpu_id)
    if gpu_mem_total > 0 and gpu_mem_free > 2000:
        LOG.rl_trainer_logger.info(f"PDQN Reload agent of process {os.getpid()}")
        return True

    return False

if __name__ == '__main__':
    try:
        mp.set_start_method(method='spawn',force=True)  # force all the multiprocessing to 'spawn' methods
        main()
    except OSError as e:
        if "win" in sys.platform:
            logging.error(f"{e.winerror}")
        logging.error(f"errno:{e.errno} strerror:{e.strerror} filename:{e.filename} filename2:{e.filename2}")
    except BaseException as e:
        logging.exception(e.args)
        logging.exception(traceback.format_exc())
        #LOG.rl_trainer_logger.exception(traceback.print_tb(sys.exc_info()[2]))
    finally:
        kill_process()
        del os.environ['PYTHONWARNINGS']