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
from algs.psac import P_SAC
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
    "batch_size": 128,
    "per_flag": True,
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    "lr_actor": 0.0002,
    "lr_critic": 0.0002,
    "lr_alpha": 0.00002,
    "gamma": 0.9,   # q值更新系数
    "tau": 0.01,    # 软更新参数
    "clip_grad": 10,
    "zero_index_gradients": True,
    "inverting_gradients": True,
}
TRAIN = True
UPDATE_FREQ = 100
modify_change_steer=False
MODEL_PATH = os.path.join(os.getcwd(), 'out', 'model_params', 'psac_ma_net_params.pth')

def main():
    SAVE_PATH = os.path.join(os.getcwd(), 'out', 'multi_agent', 'psac', 
                         datetime.datetime.today().strftime('%Y-%m-%d_%H-%M'))
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    episode_writer = SummaryWriter(SAVE_PATH)
    random.seed(0)
    torch.manual_seed(16)

    param = deepcopy(AGENT_PARAM)
    learner = P_SAC(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                        param["tau"], param["buffer_size"], param["batch_size"], 
                        param["lr_actor"], param["lr_critic"], param["lr_alpha"],
                        param["clip_grad"], param["zero_index_gradients"],
                        param["inverting_gradients"], param["per_flag"], param["device"])
    if TRAIN and os.path.exists(MODEL_PATH):
        # load pre-trained model
        learner.load_net(MODEL_PATH, map_location=learner.device)

    process = list()
    #worker_lock = Lock()
    eval_lock = Lock()
    traj_q = Queue(maxsize=param["minimal_size"])
    eval_agent_q = Queue(maxsize=1)
    #worker_agent_q = Queue(maxsize=1)
    eval_proc = mp.Process(target=worker_mp, args=
                             (eval_lock, traj_q, eval_agent_q, deepcopy(AGENT_PARAM), 0, deepcopy(SAVE_PATH), True))
    # worker_proc = mp.Process(target=worker_mp, args=
    #                          (worker_lock, traj_q, worker_agent_q, deepcopy(AGENT_PARAM), 0, deepcopy(SAVE_PATH), False))
    #process.append(worker_proc)
    eval_proc.start()
    time.sleep(20)
    process.append(eval_proc)
    #worker_proc.start()
    #[p.start() for p in process]
    with open(os.path.join(SAVE_PATH, 'log_file.txt'),'a') as file:
        file.write(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M') + '\n')

    worker_update_count, eval_update_count = 0, 0
    episode_offset = 0
    try:
        while(True):
            # Restart worker process and evaluator process if carla failed
            # if not worker_proc.is_alive():
            #     process.remove(worker_proc)
            #     if worker_agent_q.full():
            #         worker_agent_q.get(block=True, timeout=None)
            #     worker_proc = mp.Process(target=worker_mp, args=
            #                              (worker_lock, traj_q, worker_agent_q, deepcopy(AGENT_PARAM), 0, deepcopy(SAVE_PATH), False))
            #     worker_proc.start()
            #     with open(os.path.join(SAVE_PATH, 'log_file.txt'),'a') as file:
            #         file.write(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M') + '\n')
            #     process.append(worker_proc)

            if not eval_proc.is_alive():
                process.remove(eval_proc)
                if eval_agent_q.full():
                    eval_agent_q.get(block=True, timeout=None)
                eval_proc = mp.Process(target=worker_mp, args=
                                        (eval_lock, traj_q, eval_agent_q, deepcopy(AGENT_PARAM), episode_offset, deepcopy(SAVE_PATH), True))
                eval_proc.start()
                with open(os.path.join(SAVE_PATH, 'log_file.txt'),'a') as file:
                    file.write(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M') + '\n')
                process.append(eval_proc)

            #alter the batch_size and update times according to the replay buffer size:
            #reference: https://zhuanlan.zhihu.com/p/345353294, https://arxiv.org/abs/1711.00489
            k = max(learner.replay_buffer.size()// param["minimal_size"], 1)
            learner.batch_size = k * param["batch_size"]
            if traj_q.qsize() >= learner.batch_size // 10:
                for _ in range(learner.batch_size // 10):
                    trajectory=traj_q.get(block=True,timeout=None)
                    state, next_state, action, saved_action_param, reward, done, truncated, info, offset, eval \
                        = trajectory[0], trajectory[1], trajectory[2], trajectory[3], trajectory[4], \
                        trajectory[5], trajectory[6], trajectory[7], trajectory[8], trajectory[9]
                    if eval:
                        episode_offset = offset

                    learner.store_transition(state, action, saved_action_param, reward, next_state,
                                            truncated, done, info)   
            if TRAIN and learner.replay_buffer.size()>=param["minimal_size"]:
                q_loss = learner.learn()
                worker_update_count += 1
                eval_update_count += 1
                # if not worker_agent_q.full() and worker_update_count//UPDATE_FREQ > 0:
                #     worker_lock.acquire()
                #     worker_agent_q.put((deepcopy(learner.learn_time), deepcopy(q_loss)), block=True, timeout=None)
                #     learner.save_net(os.path.join(SAVE_PATH, 'worker.pth'))
                #     worker_lock.release()
                #     worker_update_count %= UPDATE_FREQ

                if not eval_agent_q.full() and eval_update_count//(UPDATE_FREQ * 2) > 0:
                    eval_lock.acquire()
                    eval_agent_q.put((deepcopy(learner.learn_time), deepcopy(q_loss)), block=True, timeout=None)
                    learner.save_net(os.path.join(SAVE_PATH, 'eval.pth'))
                    eval_lock.release()
                    eval_update_count %= UPDATE_FREQ * 2

                if learner.learn_time > 250000:
                    learner.save_net(os.path.join(SAVE_PATH, 'ipsac_250000_net_params.pth'))
                elif learner.learn_time > 200000:
                    learner.save_net(os.path.join(SAVE_PATH, 'ipsac_200000_net_params.pth'))
                elif learner.learn_time > 150000:
                    learner.save_net(os.path.join(SAVE_PATH, 'ipsac_150000_net_params.pth'))
                elif learner.learn_time > 100000:
                    learner.save_net(os.path.join(SAVE_PATH, 'ipsac_100000_net_params.pth'))
                elif learner.learn_time > 50000:
                    learner.save_net(os.path.join(SAVE_PATH, 'ipsac_50000_net_params.pth'))
                elif learner.learn_time > 20000:
                    learner.save_net(os.path.join(SAVE_PATH, 'ipsac_20000_net_params.pth'))
                episode_writer.add_scalar('Q_loss', q_loss, learner.learn_time)
    except Exception as e:
        logging.exception(e.args)
        logging.exception(traceback.format_exc())
    except KeyboardInterrupt:
        logging.info("Premature Terminated")
    finally:
        [p.join() for p in process]
        episode_writer.close()
        learner.save_net(os.path.join(SAVE_PATH, 'ipsac_final.pth'))
        logging.info('\nDone.')

def worker_mp(lock:Lock, traj_q:Queue, agent_q:Queue, agent_param:dict, episode_offset:int, save_path:str, eval:bool):
    env = gym.make("PDQNHomoNcomIndePoHiwaySAFR2CTWN5-v0")
    time.sleep(20)
    TOTAL_EPISODE = 5000
    eval = eval
    if eval:
        episode_writer = SummaryWriter(save_path)

    param = deepcopy(agent_param)
    worker = P_SAC(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                        param["tau"], param["buffer_size"], param["batch_size"], 
                        param["lr_actor"], param["lr_critic"], param["lr_alpha"],
                        param["clip_grad"], param["zero_index_gradients"],
                        param["inverting_gradients"], param["per_flag"], param["device"])
    if TRAIN and os.path.exists(MODEL_PATH):
        worker.load_net(MODEL_PATH, map_location=worker.device)

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
                        episodes = 1000 * i + i_episode + episode_offset
                        states, _ = env.reset()
                        done, truncated = False, False
                        for actor_id in states.keys():
                            ttc[actor_id], efficiency[actor_id], comfort[actor_id], lcen[actor_id],\
                                lane_change_reward[actor_id], total_reward[actor_id], avg_reward[actor_id] = \
                                -1, -1, -1, -1, 0, 0, -4
                    
                        while not done and not truncated:
                            action_dict, actions, action_params, all_action_params={}, {}, {}, {}
                            if TRAIN and not agent_q.empty():
                                lock.acquire()
                                if eval:
                                    worker.load_net(os.path.join(save_path, 'eval.pth'), map_location=worker.device)
                                else:
                                    worker.load_net(os.path.join(save_path, 'worker.pth'), map_location=worker.device)
                                learn_time, q_loss = agent_q.get()
                                lock.release()
                                worker.learn_time=learn_time
                                if q_loss is not None:
                                    LOG.rl_trainer_logger.info(f"PSAC LEARN TIME:{learn_time}, Q_loss:{q_loss}")
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
                                    LOG.rl_trainer_logger.debug(
                                        f"\nPSAC Control In Replay Buffer: actor_id: {actor_id} action: {action}, action_parameter: {saved_action_param}")

                                    traj_q.put((state, next_state, action, saved_action_param,
                                            reward, done, truncated, info, episodes, eval), block=True, timeout=None)
                                    
                                    LOG.rl_trainer_logger.debug(
                                        f"PSAC\n"
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
                                worker.save_net(os.path.join(save_path, 'ipdqn_pre_trained.pth'))
                                            
                        if done or truncated:
                            # restart the training
                            done = False
                            truncated = False

                        # record episode results
                        if env.unwrapped._rl_switch and eval:
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

                        LOG.rl_trainer_logger.info(f"PSAC Total_steps:{env.unwrapped._total_steps} RL_control_steps:{env.unwrapped._rl_control_steps}")

                        pbar.set_postfix({
                            "episodes": f"{episodes + 1}",
                            "evaluator": f"{eval}"
                        })
                        pbar.update(1)
                    except CarlaError as e:
                        LOG.rl_trainer_logger.exception("PSAC Carla Failed, restart carla!")
        
            # restart carla to clear garbage
            env.close()
    except KeyboardInterrupt:
        logging.info("Premature Terminated")
    except BaseException:
        logging.exception(traceback.format_exc())
    finally:
        if eval:
            episode_writer.close()
        traj_q.close()
        traj_q.join_thread()
        env.close()
        logging.info('\nDone.')

def reload_agent(agent, gpu_id=0):
    if agent.device != torch.device('cpu'):
        return False
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=gpu_id)
    if gpu_mem_total > 0 and gpu_mem_free > 2000:
        LOG.rl_trainer_logger.info(f"PSAC Reload agent of process {os.getpid()}")
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