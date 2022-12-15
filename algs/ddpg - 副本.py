import random, collections
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出
        self.change_buffer = collections.deque(maxlen=capacity//10)
        self.tmp_buffer = collections.deque(maxlen=10)
        self.number = 0
        # self.all_buffer = np.zeros((1000000, 66), dtype=np.float32)
        # with open('./out/replay_buffer_test.txt', 'w') as f:
        #     pass

    def add(self, state, action, reward, next_state, truncated, done, info):
        # first compress state info, then add
        state = self._compress(state)
        next_state = self._compress(next_state)
        self.tmp_buffer.append((state, action, reward, next_state, truncated, done))
        lane_center = info["offlane"]
        reward_ttc = info["TTC"]
        if reward_ttc < -0.1:
            self.change_buffer.append((state, action, reward, next_state, truncated, done))
        if lane_center > 1.0:
            self.change_buffer.append((state, action, reward, next_state, truncated, done))
        if abs(info['lane_changing_reward']) > 0.1:
            for buf in self.tmp_buffer:
                self.change_buffer.append(buf)
        self.buffer.append((state, action, reward, next_state, truncated, done))
        reward_com = info["Comfort"]

        reward_eff = info["velocity"]
        reward_yaw = info["yaw_diff"]
        # reward_list = np.array([[reward, reward_ttc, reward_com, reward_eff, reward_lan, reward_yaw]])
        print("reward_eff: ", reward_eff)
        # print("their shapes", state, action, next_state, reward_list, truncated, done)
        # state: [1, 28], action: [1, 2], next_state: [1, 28], reward_list = [1, 6], truncated = [1, 1], done = [1, 1]
        # all: [1, 66]
        if truncated == False or truncated == 0:
            truncated = 0
        else:
            truncated = 1
        if done == False or done == 0:
            done = 0
        else:
            done = 1
        # self.save_all(state, action, next_state, reward_list, np.array([truncated]), np.array([done]))
        # with open('./out/replay_buffer_test.txt','ab') as f:
        #     np.savetxt(f, state, delimiter=',')
        #     np.savetxt(f, action, delimiter=',')
        #     np.savetxt(f, np.array([reward]), delimiter=',')
        #     np.savetxt(f, next_state, delimiter=',')
        #     np.savetxt(f, np.array([truncated]), delimiter=',')
        #     np.savetxt(f, np.array([done]), delimiter=',')
        # np.save("./pre_train/state"+str(self.number)+".npy", np.array(state))
        # np.save("./pre_train/action" + str(self.number) + ".npy", np.array(action))
        # np.save("./pre_train/reward" + str(self.number) + ".npy", np.array(reward_list))
        # np.save("./pre_train/next_state" + str(self.number) + ".npy", np.array(next_state))
        # np.save("./pre_train/truncated" + str(self.number) + ".npy", np.array([truncated]))
        # np.save("./pre_train/done" + str(self.number) + ".npy", np.array([done]))


    def save_all(self, state, action, next_state, reward_list, truncated, done):
        if self.number < 1000000:
            state_ = np.reshape(state, (-1, 1))
            action_ = np.reshape(action, (-1, 1))
            next_state_ = np.reshape(next_state, (-1, 1))
            reward_list_ = np.reshape(reward_list, (-1, 1))
            truncated_ = np.reshape(truncated, (-1, 1))
            done_ = np.reshape(done, (-1, 1))
            all_feature = np.concatenate((state_, action_, next_state_, reward_list_, truncated_, done_), axis=0)
            self.all_buffer[self.number, :] = np.squeeze(all_feature)
            self.number = self.number + 1
        if self.number == 1000000:
            np.save("./out/all_replay_buffer.npy", self.all_buffer)
            self.number = self.number + 1


    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        pri_size = min(batch_size // 4, len(self.change_buffer))
        normal_size = batch_size - pri_size
        transition = random.sample(self.buffer, normal_size)
        state, action, reward, next_state, truncated, done = zip(*transition)
        pri_transition = random.sample(self.change_buffer, pri_size)
        pri_state, pri_action, pri_reward, pri_next_state, pri_truncated, pri_done = zip(*pri_transition)
        state = np.concatenate((state, pri_state), axis=0)
        action = np.concatenate((action, pri_action), axis=0)
        reward = np.concatenate((reward, pri_reward), axis=0)
        next_state = np.concatenate((next_state, pri_next_state), axis=0)
        truncated = np.concatenate((truncated, pri_truncated), axis=0)
        done = np.concatenate((done, pri_done), axis=0)
        return state, action, reward, next_state, truncated, done

    def size(self):
        return len(self.buffer)

    def _compress(self, state):
        # print('state: ', state)
        state_left_wps = np.array(state['left_waypoints'], dtype=np.float32).reshape((1, -1))
        state_center_wps = np.array(state['center_waypoints'], dtype=np.float32).reshape((1, -1))
        state_right_wps = np.array(state['right_waypoints'], dtype=np.float32).reshape((1, -1))
        state_veh_left_front = np.array(state['vehicle_info'][0], dtype=np.float32).reshape((1, -1))
        state_veh_front = np.array(state['vehicle_info'][1], dtype=np.float32).reshape((1, -1))
        state_veh_right_front = np.array(state['vehicle_info'][2], dtype=np.float32).reshape((1, -1))
        state_veh_left_rear = np.array(state['vehicle_info'][3], dtype=np.float32).reshape((1, -1))
        state_veh_rear = np.array(state['vehicle_info'][4], dtype=np.float32).reshape((1, -1))
        state_veh_right_rear = np.array(state['vehicle_info'][5], dtype=np.float32).reshape((1, -1))
        state_ev = np.array(state['ego_vehicle'], dtype=np.float32).reshape((1, -1))

        state_ = np.concatenate((state_left_wps, state_veh_left_front, state_veh_left_rear,
                                 state_center_wps, state_veh_front, state_veh_rear,
                                 state_right_wps, state_veh_right_front, state_veh_right_rear, state_ev), axis=1)
        return state_

class Split_ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity) -> None:
        self.number = 0
        split_capacity = capacity // 8
        self.dangerous_buffer = collections.deque(maxlen=split_capacity)
        self.off_center_buffer = collections.deque(maxlen=split_capacity)
        self.low_efficiency_buffer = collections.deque(maxlen=split_capacity)
        self.on_curve_buffer = collections.deque(maxlen=split_capacity)
        self.large_steering = collections.deque(maxlen=split_capacity)
        self.large_th_br = collections.deque(maxlen=split_capacity)
        self.normal_buffer = collections.deque(maxlen=capacity)
        self.all_buffer = np.zeros((1000000, 66), dtype=np.float32)
        self.ttc_thr = -0.00001
        self.lane_thr = 0.5
        self.eff_thr = 5
        self.curve_thr = 0.1
        # with open('./out/replay_buffer_test.txt', 'w') as f:
        #     pass

    def add(self, state, action, reward, next_state, truncated, done, info):
        wps = np.array(state['waypoints'], dtype=np.float32).reshape((1, -1))
        fcurve = abs(wps[0][1] - wps[0][19])
        # first compress state info, then add
        state = self._compress(state)
        next_state = self._compress(next_state)
        # state.shape: [1, 28], action.shape: [1, 2], others are float and boolean
        reward_ttc = info["TTC"]
        reward_com = info["Comfort"]
        reward_eff = info["velocity"]
        reward_lan = info["offlane"]
        reward_yaw = info["yaw_diff"]
        print("fttc: ", reward_ttc, "flane: ", reward_lan, "feff: ", reward_eff, "fcurve: ", fcurve)
        if reward_ttc < self.ttc_thr:
            # print(fTTC)
            self.dangerous_buffer.append((state, action, np.array([[reward]]), next_state, np.array([[truncated]]), np.array([[done]])))
        elif reward_lan > self.lane_thr:
            self.off_center_buffer.append((state, action, np.array([[reward]]), next_state, np.array([[truncated]]), np.array([[done]])))
        elif reward_eff < self.eff_thr:
            self.low_efficiency_buffer.append((state, action, np.array([[reward]]), next_state, np.array([[truncated]]), np.array([[done]])))
        elif fcurve > self.curve_thr:
            self.on_curve_buffer.append((state, action, np.array([[reward]]), next_state, np.array([[truncated]]), np.array([[done]])))
        elif abs(action[0][0]) > 0.8:
            self.large_steering.append((state, action, np.array([[reward]]), next_state, np.array([[truncated]]), np.array([[done]])))
        elif abs(action[0][1]) > 0.8:
            self.large_th_br.append((state, action, np.array([[reward]]), next_state, np.array([[truncated]]), np.array([[done]])))
        else:
            self.normal_buffer.append((state, action, np.array([[reward]]), next_state, np.array([[truncated]]), np.array([[done]])))

        reward_list = np.array([[reward, reward_ttc, reward_com, reward_eff, reward_lan, reward_yaw]])
        # print("reward_eff: ", reward_eff)
        # print("their shapes", state, action, next_state, reward_list, truncated, done)
        # state: [1, 28], action: [1, 2], next_state: [1, 28], reward_list = [1, 6], truncated = [1, 1], done = [1, 1]
        # all: [1, 66]
        if truncated == False or truncated == 0:
            truncated = 0
        else:
            truncated = 1
        if done == False or done == 0:
            done = 0
        else:
            done = 1
        self.save_all(state, action, next_state, reward_list, np.array([truncated]), np.array([done]))


    def save_all(self, state, action, next_state, reward_list, truncated, done):
        if self.number < 100000:
            state_ = np.reshape(state, (-1, 1))
            action_ = np.reshape(action, (-1, 1))
            next_state_ = np.reshape(next_state, (-1, 1))
            reward_list_ = np.reshape(reward_list, (-1, 1))
            truncated_ = np.reshape(truncated, (-1, 1))
            done_ = np.reshape(done, (-1, 1))
            all_feature = np.concatenate((state_, action_, next_state_, reward_list_, truncated_, done_), axis=0)
            self.all_buffer[self.number, :] = np.squeeze(all_feature)
            self.number = self.number + 1
        if self.number == 100000:
            np.save("./out/all_replay_buffer.npy", self.all_buffer)
            self.number = self.number + 1


    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        specific_size = batch_size // 8
        normal_size = batch_size - 6 * specific_size
        dangerous_transition = random.sample(self.dangerous_buffer, specific_size)
        d_state, d_action, d_reward, d_next_state, d_truncated, d_done = zip(*dangerous_transition)
        off_center_transition = random.sample(self.off_center_buffer, specific_size)
        o_state, o_action, o_reward, o_next_state, o_truncated, o_done = zip(*off_center_transition)
        low_efficiency_transition = random.sample(self.low_efficiency_buffer, specific_size)
        l_state, l_action, l_reward, l_next_state, l_truncated, l_done = zip(*low_efficiency_transition)
        on_curve_transition = random.sample(self.on_curve_buffer, specific_size)
        c_state, c_action, c_reward, c_next_state, c_truncated, c_done = zip(*on_curve_transition)
        steer_transition = random.sample(self.large_steering, specific_size)
        s_state, s_action, s_reward, s_next_state, s_truncated, s_done = zip(*steer_transition)
        th_br_transition = random.sample(self.large_th_br, specific_size)
        t_state, t_action, t_reward, t_next_state, t_truncated, t_done = zip(*th_br_transition)
        normal_transition = random.sample(self.normal_buffer, normal_size)
        n_state, n_action, n_reward, n_next_state, n_truncated, n_done = zip(*normal_transition)

        state = np.concatenate((d_state, o_state, l_state, c_state, n_state, s_state, t_state), axis=0)
        action = np.concatenate((d_action, o_action, l_action, c_action, n_action, s_action, t_action), axis=0)
        reward = np.concatenate((d_reward, o_reward, l_reward, c_reward, n_reward, s_reward, t_reward), axis=0)
        next_state = np.concatenate((d_next_state, o_next_state, l_next_state, c_next_state, n_next_state, s_next_state, t_next_state), axis=0)
        truncated = np.concatenate((d_truncated, o_truncated, l_truncated, c_truncated, n_truncated, s_truncated, t_truncated), axis=0)
        done = np.concatenate((d_done, o_done, l_done, c_done, n_done, s_done, t_done), axis=0)

        return state, action, reward, next_state, truncated, done

    def size(self):
        return len(self.normal_buffer)

    def all_size(self):
        return len(self.dangerous_buffer), len(self.off_center_buffer), len(self.on_curve_buffer), \
               len(self.large_steering), len(self.large_th_br), len(self.low_efficiency_buffer), len(self.normal_buffer)

    def running_mean_std(self):
        d_size, o_size, c_size, l_size, n_size = self.all_size()
        uniform_size = min(d_size//8, o_size, c_size, l_size, n_size)

        mean = 0
        std = 0
        return mean, std

    def _compress(self, state):
        """return state : waypoints info+ vehicle_front info, shape: 1*22,
        first 20 elements are waypoints info, the rest are vehicle info"""
        wps = np.array(state['waypoints'], dtype=np.float32).reshape((1, -1))
        ev = np.array(state['ego_vehicle'],dtype=np.float32).reshape((1,-1))
        vf = np.array(state['vehicle_front'], dtype=np.float32).reshape((1, -1))
        state_ = np.concatenate((wps, ev, vf), axis=1)

        return state_


class offline_replay_buffer:
    """
    manually adjust the sampling of replay buffer and the replay buffer remains unchanged (1,000,000 buffers)
    """
    def __init__(self):
        file_path = "../out/all_replay_buffer.npy"
        # state: [1, 28], action: [1, 2], next_state: [1, 28], reward_list = [1, 6], truncated = [1, 1], done = [1, 1]
        # reward_list = np.array([[reward, reward_ttc, reward_com, reward_eff, reward_lan, reward_yaw]])
        self.replay_buffer = np.load(file_path, allow_pickle=True)
        self.buffer_num = self.replay_buffer.shape[0]
        # print(replay_buffer.shape)
        # split five buffer of different states: dangerous, large off-center, low-efficiency, on-curve, normal
        self.dangerous_buffer = collections.deque(maxlen=250000)
        self.off_center_buffer = collections.deque(maxlen=250000)
        self.low_efficiency_buffer = collections.deque(maxlen=250000)
        self.on_curve_buffer = collections.deque(maxlen=250000)
        self.normal_buffer = collections.deque(maxlen=1000000)

        self.ttc_thr = -0.00001
        self.lane_thr = 0.5
        self.eff_thr = 5
        self.curve_thr = 0.1
        self.split_replay_buffer()

    def size(self):
        return self.buffer_num

    def split_replay_buffer(self):
        for i in range(self.buffer_num):
            current_buffer = self.replay_buffer[i]
            fTTC = current_buffer[59]
            fLane = current_buffer[62]
            feff = current_buffer[61]
            fcurve = abs(current_buffer[1]-current_buffer[19])
            print("fttc: ", fTTC, "flane: ", fLane, "feff: ", feff, "fcurve: ", fcurve)
            if fTTC < self.ttc_thr:
                # print(fTTC)
                self.dangerous_buffer.append(current_buffer)
            elif fLane > self.lane_thr:
                self.off_center_buffer.append(current_buffer)
            elif feff < self.eff_thr:
                self.low_efficiency_buffer.append(current_buffer)
            elif fcurve > self.curve_thr:
                self.on_curve_buffer.append(current_buffer)
            else:
                self.normal_buffer.append(current_buffer)
        print("dangerous_buffer: ", len(self.dangerous_buffer), "off_center_buffer: ", len(self.off_center_buffer),
              "low_efficiency_buffer: ", len(self.low_efficiency_buffer), "on_curve_buffer: ",
              len(self.on_curve_buffer),"normal_buffer: ", len(self.normal_buffer))

    def sample(self, batch_size):
        specific_size = batch_size/6
        normal_size = batch_size-4*specific_size
        dangerous_transition = random.sample(self.dangerous_buffer, specific_size)
        dangerous_transitions = zip(*dangerous_transition)
        off_center_transition = random.sample(self.off_center_buffer, specific_size)
        off_center_transitions = zip(*off_center_transition)
        low_efficiency_transition = random.sample(self.low_efficiency_buffer, specific_size)
        low_efficiency_transitions = zip(*low_efficiency_transition)
        on_curve_transition = random.sample(self.on_curve_buffer, specific_size)
        on_curve_transitions = zip(*on_curve_transition)
        normal_transition = random.sample(self.normal_buffer, normal_size)
        normal_transitions = zip(*normal_transition)
        all_transitions = np.concatenate((dangerous_transitions, off_center_transitions, on_curve_transitions,
                                          low_efficiency_transitions, normal_transitions), axis=0)
        state, action, next_state, reward, truncated, done = all_transitions[:, :28], all_transitions[:, 28:30],
        all_transitions[:, 30:56], all_transitions[:, 56:57], all_transitions[:, -2:-1], all_transitions[:, -1:]
        return state, action, reward, next_state, truncated, done

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_bound, train=True) -> None:
        # the action bound and state_dim here are dicts
        super().__init__()
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.train = train
        # self.LaneEncoder = LaneEncoder()
        # self.layer_norm=nn.LayerNorm(128)
        # self.batch_norm=nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.2)

        self.fc1_1 = nn.Linear(state_dim['waypoints'], 64)
        self.fc1_2 = nn.Linear(state_dim['ego_vehicle'],32)
        self.fc1_3 = nn.Linear(state_dim['vehicle_front'], 32)
        # concat the first layer output and input to second layer
        self.fc2 = nn.Linear(128,128)
        self.fc_out = nn.Linear(128, 2)

        # torch.nn.init.normal_(self.fc1_1.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc1_2.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.xavier_normal_(self.fc1_1.weight.data)
        # torch.nn.init.xavier_normal_(self.fc1_2.weight.data)
        # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

    def forward(self, state):
        # state : waypoints info+ vehicle_front info, shape: batch_size*22, first 20 elements are waypoints info,
        # the rest are vehicle info
        state_wp = state[:, :self.state_dim['waypoints']]
        state_ev = state[:,-self.state_dim['vehicle_front']-self.state_dim['ego_vehicle']:-self.state_dim['vehicle_front']]
        state_vf = state[:, -self.state_dim['vehicle_front']:]
        state_wp = F.relu(self.fc1_1(state_wp))
        state_ev=F.relu((self.fc1_2(state_ev)))
        state_vf = F.relu(self.fc1_3(state_vf))
        state_ = torch.cat((state_wp,state_ev, state_vf), dim=1)
        hidden = F.relu(self.fc2(state_))
        action = torch.tanh(self.fc_out(hidden))
        # steer,throttle_brake=torch.split(out,split_size_or_sections=[1,1],dim=1)
        # steer=steer.clone()
        # throttle_brake=throttle_brake.clone()
        # steer*=self.action_bound['steer']
        # throttle=throttle_brake.clone()
        # brake=throttle_brake.clone()
        # for i in range(throttle.shape[0]):
        #     if throttle[i][0]<0:
        #         throttle[i][0]=0
        #     if brake[i][0]>0:
        #         brake[i][0]=0
        # throttle*=self.action_bound['throttle']
        # brake*=self.action_bound['brake']

        return action

class veh_lane_encoder(torch.nn.Module):
    def __init__(self, state_dim, train=True):
        super().__init__()
        self.state_dim = state_dim
        self.train = train
        self.lane_encoder = nn.Linear(state_dim['waypoints'], 32)
        self.veh_encoder = nn.Linear(state_dim['conventional_vehicle'] * 2, 32)
        self.agg = nn.Linear(64, 64)

    def forward(self, lane_veh):
        lane = lane_veh[:, :self.state_dim["waypoints"]]
        veh = lane_veh[:, self.state_dim["waypoints"]:]
        lane_enc = F.relu(self.lane_encoder(lane))
        veh_enc = F.relu(self.veh_encoder(veh))
        state_cat = torch.cat((lane_enc, veh_enc), dim=1)
        state_enc = F.relu(self.agg(state_cat))
        return state_enc


class PolicyNet_multi(torch.nn.Module):
    def __init__(self, state_dim, action_bound, train=True) -> None:
        # the action bound and state_dim here are dicts
        super().__init__()
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.train = train
        self.left_encoder = veh_lane_encoder(self.state_dim)
        self.center_encoder = veh_lane_encoder(self.state_dim)
        self.right_encoder = veh_lane_encoder(self.state_dim)
        self.ego_encoder = nn.Linear(self.state_dim['ego_vehicle'], 64)
        self.fc = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 2)
        # torch.nn.init.normal_(self.fc1_1.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc1_2.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.xavier_normal_(self.fc1_1.weight.data)
        # torch.nn.init.xavier_normal_(self.fc1_2.weight.data)
        # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

    def forward(self, state):
        # state: (waypoints + 2 * conventional_vehicle0 * 3
        one_state_dim = self.state_dim['waypoints'] + self.state_dim['conventional_vehicle'] * 2
        left_enc = self.left_encoder(state[:, :one_state_dim])
        center_enc = self.center_encoder(state[:, one_state_dim:2*one_state_dim])
        right_enc = self.right_encoder(state[:, 2*one_state_dim:3*one_state_dim])
        ego_enc = self.ego_encoder(state[:, 3*one_state_dim:])
        state_ = torch.cat((left_enc, center_enc, right_enc, ego_enc), dim=1)
        hidden = F.relu(self.fc(state_))
        action = torch.tanh(self.fc_out(hidden))
        # steer,throttle_brake=torch.split(out,split_size_or_sections=[1,1],dim=1)
        # steer=steer.clone()
        # throttle_brake=throttle_brake.clone()
        # steer*=self.action_bound['steer']
        # throttle=throttle_brake.clone()
        # brake=throttle_brake.clone()
        # for i in range(throttle.shape[0]):
        #     if throttle[i][0]<0:
        #         throttle[i][0]=0
        #     if brake[i][0]>0:
        #         brake[i][0]=0
        # throttle*=self.action_bound['throttle']
        # brake*=self.action_bound['brake']

        return action


class QValueNet_multi(torch.nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        # parameter state_dim here is a dict
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.left_encoder = veh_lane_encoder(self.state_dim)
        self.center_encoder = veh_lane_encoder(self.state_dim)
        self.right_encoder = veh_lane_encoder(self.state_dim)
        self.ego_encoder = nn.Linear(self.state_dim['ego_vehicle'], 32)
        self.action_encoder = nn.Linear(self.action_dim, 32)
        self.fc = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

        # torch.nn.init.normal_(self.fc1.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.xavier_normal_(self.fc1.weight.data)
        # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

    def forward(self, state, action):
        one_state_dim = self.state_dim['waypoints'] + self.state_dim['conventional_vehicle'] * 2
        left_enc = self.left_encoder(state[:, :one_state_dim])
        center_enc = self.center_encoder(state[:, one_state_dim:2*one_state_dim])
        right_enc = self.right_encoder(state[:, 2*one_state_dim:3*one_state_dim])
        ego_enc = self.ego_encoder(state[:, 3*one_state_dim:])
        action_enc = self.action_encoder(action)
        state_ = torch.cat((left_enc, center_enc, right_enc, ego_enc, action_enc), dim=1)
        hidden = F.relu(self.fc(state_))
        out = self.fc_out(hidden)
        return out


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        # parameter state_dim here is a dict
        super().__init__()

        #self.state_dim = state_dim['waypoints'] + state_dim['ego_vehicle']+state_dim['vehicle_front']
        self.state_dim=state_dim

        self.action_dim = action_dim
        self.layer_norm = nn.LayerNorm(128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        #self.fc1 = nn.Linear(self.state_dim + action_dim, 64)

        self.fc1_1=nn.Linear(self.state_dim['waypoints'],32)
        self.fc1_2=nn.Linear(self.state_dim['ego_vehicle'],32)
        self.fc1_3=nn.Linear(self.state_dim['vehicle_front'],32)
        self.fc1_4=nn.Linear(self.action_dim,32)
        self.fc2=nn.Linear(128,128)
        self.fc_out = nn.Linear(128, 1)

        # torch.nn.init.normal_(self.fc1.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.xavier_normal_(self.fc1.weight.data)
        # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

    def forward(self, state, action):

        # state : waypoints info+ vehicle_front info, shape: batch_size*22, first 20 elements are waypoints info,
        # the rest are vehicle info
        state_wp = state[:, :self.state_dim['waypoints']]
        state_ev = state[:, -self.state_dim['vehicle_front']-self.state_dim['ego_vehicle']:-self.state_dim['vehicle_front']]
        state_vf = state[:, -self.state_dim['vehicle_front']:]
        state_wp=F.relu(self.fc1_1(state_wp))
        state_ev=F.relu(self.fc1_2(state_ev))
        state_vf=F.relu(self.fc1_3(state_vf))
        state_ac=F.relu(self.fc1_4(action))
        state = torch.cat((state_wp,state_ev,state_vf, state_ac), dim=1)
        hidden=F.relu(self.fc2(state))
        out = self.fc_out(hidden)

        return out

class MLP(nn.Module):
    r"""
    Construct a MLP in LaneEncoder, include a single fully-connected layer,
    followed by layer normalization and then ReLU.
    """

    def __init__(self, input_size, hidden_size=64):
        r"""
        self.norm is layer normalization.
        Args:
            input_size: the size of input layer.
            hidden_size: the size of output layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r"""
        Args:
            x: x.shape = [batch_size, n, input_size]
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        # x = self.fc2(x)
        return x


class LaneEncoder(torch.nn.Module):
    def __init__(self, waypoint_dim, hidden_size):
        super(LaneEncoder, self).__init__()
        self.waypoint_dim = waypoint_dim
        self.hidden_size = hidden_size
        self.MLP = MLP(self.waypoint_dim, self.hidden_size)

    def forward(self, waypoints):
        """
        :param waypoints: [batch_size, n, input_size]
        :return: (batch_size, n, input_size*2)
        """
        x = self.MLP(waypoints)
        batch_size, n, input_size = x.shape
        x2 = x.permute(0, 2, 1)  # [batch_size, input_size, n]
        x2 = F.max_pool1d(x2, kernel_size=x2.shape[2])  # [batch_size, input_size, 1]
        x2 = torch.cat([x2]*n, dim=2)  # [batch_size, input_size, n]
        y = torch.cat((x2.permute(0, 2, 1), x), dim=2)  # [batch_size, n, input_size*2]
        return y


class DDPG:
    def __init__(self, state_dim, action_dim, action_bound, gamma, tau, sigma, theta, epsilon,
                 buffer_size, batch_size, actor_lr, critic_lr, clip_grad, device) -> None:
        self.learn_time = 0
        self.replace_a = 0
        self.replace_c = 0
        self.s_dim = state_dim  # state_dim here is a dict
        self.s_dim['waypoints'] *= 3  # 2 is the feature dim of each waypoint
        self.a_dim, self.a_bound = action_dim, action_bound
        self.theta = theta
        self.gamma, self.tau, self.sigma, self.epsilon = gamma, tau, sigma, epsilon  # sigma:高斯噪声的标准差，均值直接设置为0
        self.buffer_size, self.batch_size, self.device = buffer_size, batch_size, device
        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.clip_grad = clip_grad
        # adjust different types of replay buffer
        #self.replay_buffer = Split_ReplayBuffer(buffer_size)
        self.replay_buffer = ReplayBuffer(buffer_size)
        # self.replay_buffer = offline_replay_buffer()
        """self.memory=torch.tensor((buffer_size,self.s_dim*2+self.a_dim+1+1),
            dtype=torch.float32).to(self.device)"""
        self.pointer = 0  # serve as updating the memory data
        self.train = True
        self.actor = PolicyNet_multi(self.s_dim, self.a_bound).to(self.device)
        self.actor_target = PolicyNet_multi(self.s_dim, self.a_bound).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = QValueNet_multi(self.s_dim, self.a_dim).to(self.device)
        self.critic_target = QValueNet_multi(self.s_dim, self.a_dim).to(self.device)
        # self.actor = PolicyNet(self.s_dim, self.a_bound).to(self.device)
        # self.actor_target = PolicyNet(self.s_dim, self.a_bound).to(self.device)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic = QValueNet(self.s_dim, self.a_dim).to(self.device)
        # self.critic_target = QValueNet(self.s_dim, self.a_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.loss = nn.MSELoss()

        self.steer_noise = OrnsteinUhlenbeckActionNoise(self.sigma, self.theta)
        self.tb_noise = OrnsteinUhlenbeckActionNoise(self.sigma, self.theta)

    def take_action(self, state):
        # print('vehicle_info', state['vehicle_info'])
        state_left_wps = torch.tensor(state['left_waypoints'], dtype=torch.float32).view(1, -1).to(self.device)
        state_center_wps = torch.tensor(state['center_waypoints'], dtype=torch.float32).view(1, -1).to(self.device)
        state_right_wps = torch.tensor(state['right_waypoints'], dtype=torch.float32).view(1, -1).to(self.device)
        state_veh_left_front = torch.tensor(state['vehicle_info'][0], dtype=torch.float32).view(1, -1).to(self.device)
        state_veh_front = torch.tensor(state['vehicle_info'][1], dtype=torch.float32).view(1, -1).to(self.device)
        state_veh_right_front = torch.tensor(state['vehicle_info'][2], dtype=torch.float32).view(1, -1).to(self.device)
        state_veh_left_rear = torch.tensor(state['vehicle_info'][3], dtype=torch.float32).view(1, -1).to(self.device)
        state_veh_rear = torch.tensor(state['vehicle_info'][4], dtype=torch.float32).view(1, -1).to(self.device)
        state_veh_right_rear = torch.tensor(state['vehicle_info'][5], dtype=torch.float32).view(1, -1).to(self.device)
        state_ev = torch.tensor(state['ego_vehicle'],dtype=torch.float32).view(1,-1).to(self.device)
        state_ = torch.cat((state_left_wps, state_veh_left_front, state_veh_left_rear,
                            state_center_wps, state_veh_front, state_veh_rear,
                            state_right_wps, state_veh_right_front, state_veh_right_rear, state_ev), dim=1)
        # print(state_.shape)
        action = self.actor(state_)
        print(f'Network Output - Steer: {action[0][0]}, Throttle_brake: {action[0][1]}')
        if (action[0, 0].is_cuda):
            action = np.array([action[:, 0].detach().cpu().numpy(), action[:, 1].detach().cpu().numpy()]).reshape((-1, 2))
        else:
            action = np.array([action[:, 0].detach().numpy(), action[:, 1].detach().numpy()]).reshape((-1, 2))
        # if np.random.random()<self.epsilon:
        if self.train:
            action[:, 0] = np.clip(np.random.normal(action[:, 0], self.sigma), -1, 1)
            action[:, 1] = np.clip(np.random.normal(action[:, 1], self.sigma), -1, 1)
        # if self.train:
        #     action[:,0]=np.clip(action[:,0]+self.steer_noise(),-1,1)
        #     action[:,1]=np.clip(action[:,1]+self.tb_noise(),-1,1)
        print(f'After noise - Steer: {action[0][0]}, Throttle_brake: {action[0][1]}')
        # for i in range(action.shape[0]):
        #     if action[i,1]>0:
        #         action[i,1]+=np.clip(np.random.normal(action[i,1],self.sigma),0,self.a_bound['throttle'])
        #     elif action[i,2]<0:
        #         action[i,2]+=np.clip(np.random.normal(action[i,2],self.sigma),-self.a_bound['brake'],0)

        return action


    def learn(self):
        self.learn_time += 1
        # if self.learn_time > 100000:
        #     self.train = False
        self.replace_a += 1
        self.replace_c += 1
        b_s, b_a, b_r, b_ns, b_t, b_d = self.replay_buffer.sample(self.batch_size)
        # 此处得到的batch是否是pytorch.tensor?
        batch_s = torch.tensor(b_s, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_ns = torch.tensor(b_ns, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_a = torch.tensor(b_a, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_r = torch.tensor(b_r, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_d = torch.tensor(b_d, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_t = torch.tensor(b_t, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        # compute the target Q value using the information of next state
        action_target = self.actor_target(batch_ns)
        next_q_values = self.critic_target(batch_ns, action_target)
        q_targets = batch_r + self.gamma * next_q_values * (1 - batch_t)
        q_ = self.critic(batch_s, batch_a)
        critic_loss = self.loss(q_, q_targets)
        td_max = 0
        index = 0
        for i in range(self.batch_size):
            if abs(q_[i]-q_targets[i]) > td_max:
                td_max = abs(q_[i]-q_targets[i])
                index = i
        print(f'TD-error:{critic_loss}', td_max, index)
        # print(batch_s[index], batch_ns[index], batch_a[index], batch_r[index], batch_t[index])
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optimizer.step()

        action = self.actor(batch_s)
        q = self.critic(batch_s, action)
        actor_loss = -torch.mean(q)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def _print_grad(self, model):
        '''Print the grad of each layer'''
        for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.steer_noise.set_sigma(sigma)
        self.tb_noise.set_sigma(sigma)

    def reset_noise(self):
        self.steer_noise.reset()
        self.tb_noise.reset()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, net, target_net):
        net.load_state_dict(target_net.state_dict())

    def store_transition(self, transition_dict):  # how to store the episodic data to buffer
        index = self.pointer % self.buffer_size
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float32).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float32).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float32).to(self.device)
        states_next = torch.tensor(transition_dict['states_next'],
                                   dtype=torch.float32).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float32).to(self.device)
        return

    def save_net(self,file='./out/ddpg_final.pth'):
        state = {
            'actor': self.actor.state_dict(),
            'actor_target':self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target':self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(state, file)

    def load_net(self, state):
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, sigma, theta=0.001, mu=np.array([0.0]), dt=1e-2, x0=None):
        """
        mu: The mean value of action
        theta: The bigger the value, the faster noise get close to mu
        sigma: Noise amplifier, the bigger the value, the more amplification of the disturbance
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

