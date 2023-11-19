import torch
import logging
import datetime,time, os
import gym, macad_gym
import queue
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
from multiprocessing.managers import SyncManager
sys.path.append(os.getcwd())
from main.util.process import (kill_process, get_child_processes, 
                               kill_process_and_children)
from main.util.utils import (get_gpu_info, get_gpu_mem_info)
from macad_gym.viz.logger import Logger
from macad_gym.core.simulator.carla_provider import CarlaError
from macad_gym.core.utils.wrapper import (SpeedState, Truncated)
from algs.sac_multi_lane import SACContinuous
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
    "Kaiming_normal": False,
    "buffer_size": 160000,
    "minimal_size": 10000,
    "batch_size": 256,
    "per_flag": True,
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    "lr_actor": 0.0002,
    "lr_critic": 0.0002,
    "lr_alpha": 0.00002,
    "gamma": 0.9,   # q值更新系数
    "tau": 0.01,    # 软更新参数
}
TRAIN = True
UPDATE_FREQ = 500
WORKER_NUMBER = 3
MODEL_PATH = os.path.join(os.getcwd(), 'out', 'model_params', 'sac_ma_net_params.pth')

def main():
    SAVE_PATH = os.path.join(os.getcwd(), 'out', 'multi_agent', 'sac', 
                         datetime.datetime.today().strftime('%Y-%m-%d_%H-%M'))
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    episode_writer = SummaryWriter(SAVE_PATH)
    logger = Logger('SAC trainer', os.path.join(SAVE_PATH, 'logger.txt'), logging.DEBUG, logging.WARNING)
    random.seed(0)
    torch.manual_seed(16)

    param = deepcopy(AGENT_PARAM)
    learner = SACContinuous(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                            param["tau"], param["buffer_size"], param["batch_size"], 
                            param["lr_alpha"], param["lr_actor"], param["lr_critic"], 
                            param["per_flag"], param["device"])
    if TRAIN and os.path.exists(MODEL_PATH):
        # load pre-trained model
        learner.load_net(MODEL_PATH, map_location=learner.device)

    process = list()
    worker_proc, eval_proc = [None for i in range(WORKER_NUMBER)], None
    # worker_lock, eval_lock = [None for i in range(WORKER_NUMBER)], None
    # traj_q = Queue(maxsize=param["buffer_size"])
    # worker_agent_q = [Queue(maxsize=1) for i in range(WORKER_NUMBER)]
    worker_update_count, eval_update_count = 0, 0
    episode_offset = 0

    # start a manager for multiprocess value sharing
    manager = SyncManager()
    manager.start()
    traj_q = manager.Queue(maxsize=param["buffer_size"])
    #traj_q = [manager.Queue(maxsize=param["minimal_size"]) for i in range(WORKER_NUMBER + 1)]
    worker_agent_q, eval_agent_q = [None for i in range(WORKER_NUMBER)], None

    try:
        while(True):
            # Restart worker process and evaluator process if carla failed
            def restart_eval(eval_proc, eval_agent_q, episode_offset):
                agent_q = eval_agent_q
                if not eval_proc or not eval_proc.is_alive():
                    if eval_proc:
                        process.remove(eval_proc)
                    agent_q = manager.Queue(maxsize=1)
                    eval_proc = mp.Process(target=worker_mp, args=
                                            (traj_q, agent_q, deepcopy(AGENT_PARAM), episode_offset, deepcopy(SAVE_PATH), -1))
                    eval_proc.start()
                    with open(os.path.join(SAVE_PATH, 'log_file.txt'),'a') as file:
                        file.write(f"{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')} evaluator \n")
                    process.append(eval_proc)
                    time.sleep(60)
                
                return eval_proc, agent_q
            
            def restart_worker(worker_proc, worker_agent_q, index):
                agent_q = worker_agent_q
                if not worker_proc or not worker_proc.is_alive():
                    if worker_proc:
                        process.remove(worker_proc)
                    agent_q = manager.Queue(maxsize=1)
                    worker_proc = mp.Process(target=worker_mp, args=
                                            (traj_q, agent_q, deepcopy(AGENT_PARAM), 0, deepcopy(SAVE_PATH), index))
                    worker_proc.start()
                    with open(os.path.join(SAVE_PATH, 'log_file.txt'),'a') as file:
                        file.write(f"{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')} worker_{i} \n")
                    process.append(worker_proc)
                    time.sleep(60)
                
                return worker_proc, agent_q

            eval_proc, eval_agent_q = restart_eval(eval_proc, eval_agent_q, episode_offset)

            for i in range(WORKER_NUMBER):
                worker_proc[i], worker_agent_q[i] = restart_worker(worker_proc[i], worker_agent_q[i], i)

            #alter the batch_size and update times according to the replay buffer size:
            #reference: https://zhuanlan.zhihu.com/p/345353294, https://arxiv.org/abs/1711.00489
            k = max(learner.replay_buffer.size()// param["minimal_size"], 1)
            learner.batch_size = k * param["batch_size"]
            for i in range(WORKER_NUMBER + 1):
                if traj_q.qsize() >= learner.batch_size // (WORKER_NUMBER * 2):
                    for _ in range(learner.batch_size // (WORKER_NUMBER * 2)):
                        trajectory=traj_q.get(block=True,timeout=None)
                        state, next_state, action, reward, done, truncated, info, offset, eval= \
                            trajectory[0], trajectory[1], trajectory[2], trajectory[3], \
                            trajectory[4], trajectory[5], trajectory[6], trajectory[7], trajectory[8]
                        if eval:
                            episode_offset = offset

                        learner.store_transition(state, action, reward, next_state,
                                                truncated, done, info)       
            if TRAIN and learner.replay_buffer.size()>=param["minimal_size"]:
                q_loss = learner.learn()
                worker_update_count += 1
                eval_update_count += 1

                if eval_update_count // UPDATE_FREQ > 0:
                    if not eval_agent_q.full():
                        learner.save_net(os.path.join(SAVE_PATH, 'eval.pth'))
                    try:
                        eval_agent_q.put((deepcopy(learner.learn_time), deepcopy(q_loss)), block=True, timeout=10)
                    except queue.Full as e:
                        logger.exception(f"SAC put to full agent_q of evaluator, {e.args}")
                        kill_process_and_children(logger, eval_proc.pid)
                        eval_proc, eval_agent_q = restart_eval(eval_proc, eval_agent_q, episode_offset)
                        continue
                    #eval_lock.release()
                    eval_update_count %= UPDATE_FREQ

                if worker_update_count // (UPDATE_FREQ * 2)> 0:
                    for i in range(WORKER_NUMBER):
                        if not worker_agent_q[i].full():
                            learner.save_net(os.path.join(SAVE_PATH, f'worker_{i}.pth'))
                        try:
                            worker_agent_q[i].put((deepcopy(learner.learn_time), deepcopy(q_loss)), block=True, timeout=10)
                        except queue.Full as e:
                            logger.exception(f"SAC put to full agnet_q of worker_{i}, {e.args}")
                            kill_process_and_children(logger, worker_proc[i].pid)
                            worker_proc[i], worker_agent_q[i] = restart_worker(worker_proc[i], worker_agent_q[i], i)
                            continue
                    
                    worker_update_count %= UPDATE_FREQ * 2

                if learner.learn_time >= 50000 and learner.learn_time % 50000 == 0:
                    learner.save_net(os.path.join(SAVE_PATH, f'isac_{learner.learn_time}_net_params.pth'))
                elif learner.learn_time > 20000:
                    learner.save_net(os.path.join(SAVE_PATH, 'isac_20000_net_params.pth'))
                episode_writer.add_scalar('Q_loss', q_loss, learner.learn_time)
    except KeyboardInterrupt:
        logger.info("Premature Terminated")
    except Exception as e:
        logger.exception(e.args)
        logger.exception(traceback.format_exc())
    finally:
        [p.join() for p in process]
        episode_writer.close()
        manager.shutdown()
        learner.save_net(os.path.join(SAVE_PATH, 'isac_final.pth'))
        logger.info('\nDone.')

def worker_mp(traj_q:queue.Queue, agent_q:queue.Queue, param:dict, episode_offset:int, save_path:str, index:int):
    env = gym.make("HomoNcomIndePoHiwaySAFR2CTWN5-v0")
    TOTAL_EPISODE = 1000
    if index == -1:
        # This process is evaluator
        eval = True
        env.unwrapped.pre_train_steps = 0
        episode_writer = SummaryWriter(save_path)
    else:
        eval = False
        if index != 0:
            param["device"] = torch.device('cpu')
    logger = Logger(f"SAC {'evaluator' if eval else 'worker_'+str(index)}", 
                    os.path.join(save_path, 'logger.txt'), logging.DEBUG, logging.WARNING)

    worker = SACContinuous(param["s_dim"], param["a_dim"], param["a_bound"], param["gamma"],
                            param["tau"], param["buffer_size"], param["batch_size"], 
                            param["lr_alpha"], param["lr_actor"], param["lr_critic"], 
                            param["per_flag"], param["device"])
    if eval and os.path.exists(os.path.join(save_path, 'eval.pth')):
        # make sure the evaluator excute the newest model
        worker.load_net(os.path.join(save_path, 'eval.pth'), map_location=worker.device)
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
        for i in range(TOTAL_EPISODE//500):
            with tqdm(total=500, desc="Iteration %d" % i) as pbar:
                for i_episode in range(500):
                    episodes = 500 * i + i_episode + episode_offset
                    states, _ = env.reset()
                    done, truncated = False, False
                    for actor_id in states.keys():
                        ttc[actor_id], efficiency[actor_id], comfort[actor_id], lcen[actor_id],\
                            lane_change_reward[actor_id], total_reward[actor_id], avg_reward[actor_id] = \
                            -1, -1, -1, -1, 0, 0, -4
                
                    while not done and not truncated:
                        actions = {}
                        if TRAIN and not agent_q.empty():
                            #lock.acquire()
                            if eval:
                                worker.load_net(os.path.join(save_path, 'eval.pth'), map_location=worker.device)
                            else:
                                worker.load_net(os.path.join(save_path, f'worker_{index}.pth'), map_location=worker.device)
                            try:
                                learn_time, q_loss = agent_q.get_nowait()
                            except queue.Empty as e:
                                logger.exception(f"SAC {'evaluator' if eval else 'worker_'+str(index)} get from empty agent_q, {e.args}")
                                continue
                            #lock.release()
                            worker.learn_time=learn_time
                            if q_loss is not None:
                                env.log(f"SAC LEARN TIME:{learn_time}, Q_loss:{q_loss}", "INFO")
                                losses_episode.append(q_loss)

                        for actor_id in states.keys():
                            actions[actor_id] = worker.take_action(states[actor_id][1])
                            
                        next_states, rewards, dones, truncateds, infos = env.step(actions)
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
                                action = [[info["control_info"]["steer"], throttle_brake]]
                                env.log(
                                    f"\nSAC Control In Replay Buffer: actor_id: {actor_id} action: {action}", "DEBUG")
                                
                                try:
                                    # XXX Do not set timeout=None here, otherwise the process might end in a perpetual blocked state.
                                    traj_q.put_nowait((state, next_state, action, reward, done, 
                                                       truncated, info, episodes, eval))
                                except queue.Full as e:
                                    logger.exception(f"SAC {'evaluator' if eval else 'worker_'+str(index)} put traj to full traj_q, {e.args}")
                                    continue
                                
                                env.log(
                                    f"SAC\n"
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
                                    f"network_action: {actions[actor_id]}, saved_actions: {action} \n"
                                    f"reward: {reward}, truncated: {truncated}, done: {done}, ",
                                    "DEBUG")
    
                        done = dones["__all__"]
                        truncated = truncateds["__all__"]!=Truncated.FALSE
                        states = next_states
                        
                        #only record the first vehicle reward
                        if env.unwrapped._total_steps == env.unwrapped.pre_train_steps:
                            worker.save_net(os.path.join(save_path, 'isac_pre_trained.pth'))
                    
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

                    env.log(f"SAC Total_steps:{env.unwrapped._total_steps} RL_control_steps:{env.unwrapped._rl_control_steps}", "INFO")

                    pbar.set_postfix({
                        "episodes": f"{episodes + 1}",
                        "evaluator": f"{True if eval else 'worker_'+str(index)}"
                    })
                    pbar.update(1)      
        
            # restart carla to clear garbage
            env.close()
    except KeyboardInterrupt:
        logger.info("SAC Premature Terminated")
    except CarlaError as e:
        logger.exception(f"SAC {'evaluator' if eval else 'worker'} error due to Carla failure, terminate process!")
    except BaseException:
        logger.exception(f"SAC {traceback.format_exc()}")
    finally:
        if eval:
            episode_writer.close()
        logger.info(f"SAC Exit {'evaluator' if eval else 'worker_'+str(index)} process")
        sys.exit(1)

def reload_agent(agent, gpu_id=0):
    if agent.device != torch.device('cpu'):
        return False
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=gpu_id)
    if gpu_mem_total > 0 and gpu_mem_free > 2000:
        print(f"SAC Reload agent of process {os.getpid()}")
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
    finally:
        kill_process()
        del os.environ['PYTHONWARNINGS']