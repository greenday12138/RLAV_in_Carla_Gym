import random, collections
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# class ReplayBuffer:
#     """经验回放池"""

#     def __init__(self, capacity) -> None:
#         self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出
#         self.change_buffer = collections.deque(maxlen=capacity//10)
#         self.tmp_buffer = collections.deque(maxlen=10)
#         self.number = 0
#         # self.all_buffer = np.zeros((1000000, 66), dtype=np.float32)
#         # with open('./out/replay_buffer_test.txt', 'w') as f:
#         #     pass

#     def add(self, state, action, reward, next_state, truncated, done, info):
#         # first compress state info, then add
#         state = self._compress(state)
#         next_state = self._compress(next_state)
#         self.tmp_buffer.append((state, action, reward, next_state, truncated, done))
#         lane_center = info["offlane"]
#         reward_ttc = info["TTC"]
#         if reward_ttc < -0.1:
#             self.change_buffer.append((state, action, reward, next_state, truncated, done))
#         if lane_center > 1.0:
#             self.change_buffer.append((state, action, reward, next_state, truncated, done))
#         if abs(info['lane_changing_reward']) > 0.1:
#             for buf in self.tmp_buffer:
#                 self.change_buffer.append(buf)
#         self.buffer.append((state, action, reward, next_state, truncated, done))
#         reward_com = info["Comfort"]

#         reward_eff = info["velocity"]
#         reward_yaw = info["yaw_diff"]
#         # reward_list = np.array([[reward, reward_ttc, reward_com, reward_eff, reward_lan, reward_yaw]])
#         print("reward_eff: ", reward_eff)
#         # print("their shapes", state, action, next_state, reward_list, truncated, done)
#         # state: [1, 28], action: [1, 2], next_state: [1, 28], reward_list = [1, 6], truncated = [1, 1], done = [1, 1]
#         # all: [1, 66]
#         if truncated == False or truncated == 0:
#             truncated = 0
#         else:
#             truncated = 1
#         if done == False or done == 0:
#             done = 0
#         else:
#             done = 1


#     def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
#         pri_size = min(batch_size // 4, len(self.change_buffer))
#         normal_size = batch_size - pri_size
#         transition = random.sample(self.buffer, normal_size)
#         state, action, reward, next_state, truncated, done = zip(*transition)
#         pri_transition = random.sample(self.change_buffer, pri_size)
#         pri_state, pri_action, pri_reward, pri_next_state, pri_truncated, pri_done = zip(*pri_transition)
#         state = np.concatenate((state, pri_state), axis=0)
#         action = np.concatenate((action, pri_action), axis=0)
#         reward = np.concatenate((reward, pri_reward), axis=0)
#         next_state = np.concatenate((next_state, pri_next_state), axis=0)
#         truncated = np.concatenate((truncated, pri_truncated), axis=0)
#         done = np.concatenate((done, pri_done), axis=0)
#         return state, action, reward, next_state, truncated, done

#     def size(self):
#         return len(self.buffer)

#     def _compress(self, state):
#         # print('state: ', state)
#         state_left_wps = np.array(state['left_waypoints'], dtype=np.float32).reshape((1, -1))
#         state_center_wps = np.array(state['center_waypoints'], dtype=np.float32).reshape((1, -1))
#         state_right_wps = np.array(state['right_waypoints'], dtype=np.float32).reshape((1, -1))
#         state_veh_left_front = np.array(state['vehicle_info'][0], dtype=np.float32).reshape((1, -1))
#         state_veh_front = np.array(state['vehicle_info'][1], dtype=np.float32).reshape((1, -1))
#         state_veh_right_front = np.array(state['vehicle_info'][2], dtype=np.float32).reshape((1, -1))
#         state_veh_left_rear = np.array(state['vehicle_info'][3], dtype=np.float32).reshape((1, -1))
#         state_veh_rear = np.array(state['vehicle_info'][4], dtype=np.float32).reshape((1, -1))
#         state_veh_right_rear = np.array(state['vehicle_info'][5], dtype=np.float32).reshape((1, -1))
#         state_ev = np.array(state['ego_vehicle'], dtype=np.float32).reshape((1, -1))

#         state_ = np.concatenate((state_left_wps, state_veh_left_front, state_veh_left_rear,
#                                  state_center_wps, state_veh_front, state_veh_rear,
#                                  state_right_wps, state_veh_right_front, state_veh_right_rear, state_ev), axis=1)
#         return state_

class ReplayBuffer:

    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出
        self.change_buffer = collections.deque(maxlen=capacity//10)
        self.tmp_buffer = collections.deque(maxlen=10)
        self.number = 0

    def add(self, state, action, action_param, reward, next_state, truncated, done, info):
        # first compress state info, then add
        state = self._compress(state)
        next_state = self._compress(next_state)
        if not truncated:
            lane_center = info["offlane"]
            reward_ttc = info["TTC"]
            reward_eff = info["velocity"]
            reward_com = info["Comfort"]
            reward_eff = info["velocity"]
            reward_yaw = info["yaw_diff"]
        # if reward_ttc < -0.1 or reward_eff < 3:
        #     self.change_buffer.append((state, action, action_param, reward, next_state, truncated, done))
        # if truncated:
        #     self.change_buffer.append((state, action, action_param, reward, next_state, truncated, done))
        if action == 0 or action == 2:
            self.change_buffer.append((state, action, action_param, reward, next_state, truncated, done))
        self.tmp_buffer.append((state, action, action_param, reward, next_state, truncated, done))
        # if info['lane_changing_reward'] > 0.1:
        #     for buf in self.tmp_buffer:
        #         self.change_buffer.append(buf)
        self.buffer.append((state, action, action_param, reward, next_state, truncated, done))
        # print("their shapes", state, action, next_state, reward_list, truncated, done)
        # state: [1, 28], action: [1, 2], next_state: [1, 28], reward_list = [1, 6], truncated = [1, 1], done = [1, 1]
        # all: [1, 66]

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        pri_size = min(batch_size // 2, len(self.change_buffer))
        normal_size = batch_size - pri_size
        transition = random.sample(self.buffer, normal_size)
        state, action, action_param, reward, next_state, truncated, done = zip(*transition)
        pri_transition = random.sample(self.change_buffer, pri_size)
        pri_state, pri_action, pri_action_param, pri_reward, pri_next_state, pri_truncated, pri_done = zip(*pri_transition)
        state = np.concatenate((state, pri_state), axis=0)
        action = np.concatenate((action, pri_action), axis=0)
        action_param = np.concatenate((action_param, pri_action_param), axis=0)
        reward = np.concatenate((reward, pri_reward), axis=0)
        next_state = np.concatenate((next_state, pri_next_state), axis=0)
        truncated = np.concatenate((truncated, pri_truncated), axis=0)
        done = np.concatenate((done, pri_done), axis=0)
        return state, action, action_param, reward, next_state, truncated, done

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
        state_light = np.array(state['light'], dtype=np.float32).reshape((1, -1))
        state_ = np.concatenate((state_left_wps, state_veh_left_front, state_veh_left_rear, state_light,
                                 state_center_wps, state_veh_front, state_veh_rear, state_light,
                                 state_right_wps, state_veh_right_front, state_veh_right_rear, state_light, state_ev), axis=1)
        return state_

# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, action_bound, train=True) -> None:
#         # the action bound and state_dim here are dicts
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_bound = action_bound
#         self.train = train
#         # self.LaneEncoder = LaneEncoder()
#         # self.layer_norm=nn.LayerNorm(128)
#         # self.batch_norm=nn.BatchNorm2d(128)
#         self.dropout = nn.Dropout(0.2)

#         self.fc1_1 = nn.Linear(state_dim['waypoints'], 64)
#         self.fc1_2 = nn.Linear(state_dim['ego_vehicle'],32)
#         self.fc1_3 = nn.Linear(state_dim['vehicle_front'], 32)
#         # concat the first layer output and input to second layer
#         self.fc2 = nn.Linear(128,128)
#         self.fc_out = nn.Linear(128, 2)

#         # torch.nn.init.normal_(self.fc1_1.weight.data,0,0.01)
#         # torch.nn.init.normal_(self.fc1_2.weight.data,0,0.01)
#         # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
#         # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
#         # torch.nn.init.xavier_normal_(self.fc1_1.weight.data)
#         # torch.nn.init.xavier_normal_(self.fc1_2.weight.data)
#         # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

#     def forward(self, state):
#         # state : waypoints info+ vehicle_front info, shape: batch_size*22, first 20 elements are waypoints info,
#         # the rest are vehicle info
#         state_wp = state[:, :self.state_dim['waypoints']]
#         state_ev = state[:,-self.state_dim['vehicle_front']-self.state_dim['ego_vehicle']:-self.state_dim['vehicle_front']]
#         state_vf = state[:, -self.state_dim['vehicle_front']:]
#         state_wp = F.relu(self.fc1_1(state_wp))
#         state_ev=F.relu((self.fc1_2(state_ev)))
#         state_vf = F.relu(self.fc1_3(state_vf))
#         state_ = torch.cat((state_wp,state_ev, state_vf), dim=1)
#         hidden = F.relu(self.fc2(state_))
#         action = torch.tanh(self.fc_out(hidden))
#         # steer,throttle_brake=torch.split(out,split_size_or_sections=[1,1],dim=1)
#         # steer=steer.clone()
#         # throttle_brake=throttle_brake.clone()
#         # steer*=self.action_bound['steer']
#         # throttle=throttle_brake.clone()
#         # brake=throttle_brake.clone()
#         # for i in range(throttle.shape[0]):
#         #     if throttle[i][0]<0:
#         #         throttle[i][0]=0
#         #     if brake[i][0]>0:
#         #         brake[i][0]=0
#         # throttle*=self.action_bound['throttle']
#         # brake*=self.action_bound['brake']

#         return action

# class veh_lane_encoder(torch.nn.Module):
#     def __init__(self, state_dim, train=True):
#         super().__init__()
#         self.state_dim = state_dim
#         self.train = train
#         self.lane_encoder = nn.Linear(state_dim['waypoints'], 32)
#         self.veh_encoder = nn.Linear(state_dim['companion_vehicle'] * 2, 32)
#         self.agg = nn.Linear(64, 64)

#     def forward(self, lane_veh):
#         lane = lane_veh[:, :self.state_dim["waypoints"]]
#         veh = lane_veh[:, self.state_dim["waypoints"]:]
#         lane_enc = F.relu(self.lane_encoder(lane))
#         veh_enc = F.relu(self.veh_encoder(veh))
#         state_cat = torch.cat((lane_enc, veh_enc), dim=1)
#         state_enc = F.relu(self.agg(state_cat))
#         return state_enc


# class PolicyNet_multi(torch.nn.Module):
#     def __init__(self, state_dim, action_bound, train=True) -> None:
#         # the action bound and state_dim here are dicts
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_bound = action_bound
#         self.train = train
#         self.left_encoder = veh_lane_encoder(self.state_dim)
#         self.center_encoder = veh_lane_encoder(self.state_dim)
#         self.right_encoder = veh_lane_encoder(self.state_dim)
#         self.ego_encoder = nn.Linear(self.state_dim['ego_vehicle'], 64)
#         self.fc = nn.Linear(256, 256)
#         self.fc_out = nn.Linear(256, 2)
#         # torch.nn.init.normal_(self.fc1_1.weight.data,0,0.01)
#         # torch.nn.init.normal_(self.fc1_2.weight.data,0,0.01)
#         # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
#         # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
#         # torch.nn.init.xavier_normal_(self.fc1_1.weight.data)
#         # torch.nn.init.xavier_normal_(self.fc1_2.weight.data)
#         # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

#     def forward(self, state):
#         # state: (waypoints + 2 * companion_vehicle * 3
#         one_state_dim = self.state_dim['waypoints'] + self.state_dim['companion_vehicle'] * 2
#         left_enc = self.left_encoder(state[:, :one_state_dim])
#         center_enc = self.center_encoder(state[:, one_state_dim:2*one_state_dim])
#         right_enc = self.right_encoder(state[:, 2*one_state_dim:3*one_state_dim])
#         ego_enc = self.ego_encoder(state[:, 3*one_state_dim:])
#         state_ = torch.cat((left_enc, center_enc, right_enc, ego_enc), dim=1)
#         hidden = F.relu(self.fc(state_))
#         action = torch.tanh(self.fc_out(hidden))
#         # steer,throttle_brake=torch.split(out,split_size_or_sections=[1,1],dim=1)
#         # steer=steer.clone()
#         # throttle_brake=throttle_brake.clone()
#         # steer*=self.action_bound['steer']
#         # throttle=throttle_brake.clone()
#         # brake=throttle_brake.clone()
#         # for i in range(throttle.shape[0]):
#         #     if throttle[i][0]<0:
#         #         throttle[i][0]=0
#         #     if brake[i][0]>0:
#         #         brake[i][0]=0
#         # throttle*=self.action_bound['throttle']
#         # brake*=self.action_bound['brake']

#         return action


# class QValueNet_multi(torch.nn.Module):
#     def __init__(self, state_dim, action_dim) -> None:
#         # parameter state_dim here is a dict
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.left_encoder = veh_lane_encoder(self.state_dim)
#         self.center_encoder = veh_lane_encoder(self.state_dim)
#         self.right_encoder = veh_lane_encoder(self.state_dim)
#         self.ego_encoder = nn.Linear(self.state_dim['ego_vehicle'], 32)
#         self.action_encoder = nn.Linear(self.action_dim, 32)
#         self.fc = nn.Linear(256, 256)
#         self.fc_out = nn.Linear(256, 1)

#         # torch.nn.init.normal_(self.fc1.weight.data,0,0.01)
#         # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
#         # torch.nn.init.xavier_normal_(self.fc1.weight.data)
#         # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

#     def forward(self, state, action):
#         one_state_dim = self.state_dim['waypoints'] + self.state_dim['companion_vehicle'] * 2
#         left_enc = self.left_encoder(state[:, :one_state_dim])
#         center_enc = self.center_encoder(state[:, one_state_dim:2*one_state_dim])
#         right_enc = self.right_encoder(state[:, 2*one_state_dim:3*one_state_dim])
#         ego_enc = self.ego_encoder(state[:, 3*one_state_dim:])
#         action_enc = self.action_encoder(action)
#         state_ = torch.cat((left_enc, center_enc, right_enc, ego_enc, action_enc), dim=1)
#         hidden = F.relu(self.fc(state_))
#         out = self.fc_out(hidden)
#         return out


# class QValueNet(torch.nn.Module):
#     def __init__(self, state_dim, action_dim) -> None:
#         # parameter state_dim here is a dict
#         super().__init__()

#         #self.state_dim = state_dim['waypoints'] + state_dim['ego_vehicle']+state_dim['vehicle_front']
#         self.state_dim=state_dim

#         self.action_dim = action_dim
#         self.layer_norm = nn.LayerNorm(128)
#         self.batch_norm = nn.BatchNorm1d(128)
#         self.dropout = nn.Dropout(0.2)
#         #self.fc1 = nn.Linear(self.state_dim + action_dim, 64)

#         self.fc1_1=nn.Linear(self.state_dim['waypoints'],32)
#         self.fc1_2=nn.Linear(self.state_dim['ego_vehicle'],32)
#         self.fc1_3=nn.Linear(self.state_dim['vehicle_front'],32)
#         self.fc1_4=nn.Linear(self.action_dim,32)
#         self.fc2=nn.Linear(128,128)
#         self.fc_out = nn.Linear(128, 1)

#         # torch.nn.init.normal_(self.fc1.weight.data,0,0.01)
#         # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
#         # torch.nn.init.xavier_normal_(self.fc1.weight.data)
#         # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

#     def forward(self, state, action):

#         # state : waypoints info+ vehicle_front info, shape: batch_size*22, first 20 elements are waypoints info,
#         # the rest are vehicle info
#         state_wp = state[:, :self.state_dim['waypoints']]
#         state_ev = state[:, -self.state_dim['vehicle_front']-self.state_dim['ego_vehicle']:-self.state_dim['vehicle_front']]
#         state_vf = state[:, -self.state_dim['vehicle_front']:]
#         state_wp=F.relu(self.fc1_1(state_wp))
#         state_ev=F.relu(self.fc1_2(state_ev))
#         state_vf=F.relu(self.fc1_3(state_vf))
#         state_ac=F.relu(self.fc1_4(action))
#         state = torch.cat((state_wp,state_ev,state_vf, state_ac), dim=1)
#         hidden=F.relu(self.fc2(state))
#         out = self.fc_out(hidden)

#         return out

# class MLP(nn.Module):
#     r"""
#     Construct a MLP in LaneEncoder, include a single fully-connected layer,
#     followed by layer normalization and then ReLU.
#     """

#     def __init__(self, input_size, hidden_size=64):
#         r"""
#         self.norm is layer normalization.
#         Args:
#             input_size: the size of input layer.
#             hidden_size: the size of output layer.
#         """
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.norm = torch.nn.LayerNorm(hidden_size)
#         # self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         r"""
#         Args:
#             x: x.shape = [batch_size, n, input_size]
#         """
#         x = self.fc1(x)
#         x = self.norm(x)
#         x = F.relu(x)
#         # x = self.fc2(x)
#         return x


# class LaneEncoder(torch.nn.Module):
#     def __init__(self, waypoint_dim, hidden_size):
#         super(LaneEncoder, self).__init__()
#         self.waypoint_dim = waypoint_dim
#         self.hidden_size = hidden_size
#         self.MLP = MLP(self.waypoint_dim, self.hidden_size)

#     def forward(self, waypoints):
#         """
#         :param waypoints: [batch_size, n, input_size]
#         :return: (batch_size, n, input_size*2)
#         """
#         x = self.MLP(waypoints)
#         batch_size, n, input_size = x.shape
#         x2 = x.permute(0, 2, 1)  # [batch_size, input_size, n]
#         x2 = F.max_pool1d(x2, kernel_size=x2.shape[2])  # [batch_size, input_size, 1]
#         x2 = torch.cat([x2]*n, dim=2)  # [batch_size, input_size, n]
#         y = torch.cat((x2.permute(0, 2, 1), x), dim=2)  # [batch_size, n, input_size*2]
#         return y


class veh_lane_encoder(torch.nn.Module):
    def __init__(self, state_dim, train=True):
        super().__init__()
        self.state_dim = state_dim
        self.train = train
        self.lane_encoder = nn.Linear(state_dim['waypoints'], 32)
        self.veh_encoder = nn.Linear(state_dim['companion_vehicle'] * 2, 64)
        self.light_encoder = nn.Linear(state_dim['light'], 32)
        self.agg = nn.Linear(128, 64)

    def forward(self, lane_veh, ego_info):
        lane = lane_veh[:, :self.state_dim["waypoints"]]
        veh = lane_veh[:, self.state_dim["waypoints"]:-self.state_dim['light']]
        light = lane_veh[:, -self.state_dim['light']:]
        lane_enc = F.relu(self.lane_encoder(lane))
        veh_enc = F.relu(self.veh_encoder(veh))
        light_enc = F.relu(self.light_encoder(light))
        state_cat = torch.cat((lane_enc, veh_enc, light_enc), dim=1)
        state_enc = F.relu(self.agg(state_cat))
        return state_enc


class lane_wise_cross_attention_encoder(torch.nn.Module):
    def __init__(self, state_dim, train=True):
        super().__init__()
        self.state_dim = state_dim
        self.train = train
        self.hidden_size = 64
        self.lane_encoder = nn.Linear(state_dim['waypoints'], self.hidden_size)
        self.veh_encoder = nn.Linear(state_dim['companion_vehicle'] * 2, self.hidden_size)
        self.light_encoder = nn.Linear(state_dim['light'], self.hidden_size)
        self.ego_encoder = nn.Linear(state_dim['ego_vehicle'], self.hidden_size)
        self.w = nn.Linear(self.hidden_size, self.hidden_size)
        self.ego_a = nn.Linear(self.hidden_size, 1)
        self.ego_o = nn.Linear(self.hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, lane_veh, ego_info):
        batch_size = lane_veh.shape[0]
        lane = lane_veh[:, :self.state_dim["waypoints"]]
        veh = lane_veh[:, self.state_dim["waypoints"]:-self.state_dim['light']]
        light = lane_veh[:, -self.state_dim['light']:]
        # print('ego_info.shape: ', ego_info.shape)
        ego_enc = self.w(F.relu(self.ego_encoder(ego_info)))
        lane_enc = self.w(F.relu(self.lane_encoder(lane)))
        veh_enc = self.w(F.relu(self.veh_encoder(veh)))
        light_enc = self.w(F.relu(self.light_encoder(light)))
        state_enc = torch.cat((lane_enc, veh_enc, light_enc), 1).reshape(batch_size, 3, self.hidden_size)
        # _enc: [batch_size, 32]
        score_lane = self.leaky_relu(self.ego_a(ego_enc) + self.ego_o(lane_enc))
        score_veh = self.leaky_relu(self.ego_a(ego_enc) + self.ego_o(veh_enc))
        score_light = self.leaky_relu(self.ego_a(ego_enc) + self.ego_o(light_enc))
        # score_: [batch_size, 1]
        score = torch.cat((score_lane, score_veh, score_light), 1)
        score = F.softmax(score, 1).reshape(batch_size, 1, 3)
        state_enc = torch.matmul(score, state_enc).reshape(batch_size, self.hidden_size)
        # state_enc: [N, 64]
        return state_enc


class PolicyNet_multi(torch.nn.Module):
    def __init__(self, state_dim, action_parameter_size, action_bound, train=True) -> None:
        # the action bound and state_dim here are dicts
        super().__init__()
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.action_parameter_size = action_parameter_size
        self.train = train
        self.left_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.center_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.right_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.ego_encoder = nn.Linear(self.state_dim['ego_vehicle'], 64)
        self.fc = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, self.action_parameter_size)
        # torch.nn.init.normal_(self.fc1_1.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc1_2.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.xavier_normal_(self.fc1_1.weight.data)
        # torch.nn.init.xavier_normal_(self.fc1_2.weight.data)
        # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

    def forward(self, state):
        # state: (waypoints + 2 * companion_vehicle * 3
        one_state_dim = self.state_dim['waypoints'] + self.state_dim['companion_vehicle'] * 2 + self.state_dim['light']
        # print(state.shape, one_state_dim)
        ego_info = state[:, 3*one_state_dim:]
        # print(ego_info.shape)
        left_enc = self.left_encoder(state[:, :one_state_dim], ego_info)
        center_enc = self.center_encoder(state[:, one_state_dim:2*one_state_dim], ego_info)
        right_enc = self.right_encoder(state[:, 2*one_state_dim:3*one_state_dim], ego_info)
        ego_enc = self.ego_encoder(ego_info)
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
    def __init__(self, state_dim, action_param_dim, num_actions) -> None:
        # parameter state_dim here is a dict
        super().__init__()
        self.state_dim = state_dim
        self.action_param_dim = action_param_dim
        self.num_actions = num_actions
        self.left_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.center_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.right_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.ego_encoder = nn.Linear(self.state_dim['ego_vehicle'], 32)
        self.action_encoder = nn.Linear(self.action_param_dim, 32)
        self.fc = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, self.num_actions)

        # torch.nn.init.normal_(self.fc1.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.xavier_normal_(self.fc1.weight.data)
        # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

    def forward(self, state, action):
        one_state_dim = self.state_dim['waypoints'] + self.state_dim['companion_vehicle'] * 2 + self.state_dim['light']
        ego_info = state[:, 3*one_state_dim:]
        left_enc = self.left_encoder(state[:, :one_state_dim], ego_info)
        center_enc = self.center_encoder(state[:, one_state_dim:2*one_state_dim], ego_info)
        right_enc = self.right_encoder(state[:, 2*one_state_dim:3*one_state_dim], ego_info)
        ego_enc = self.ego_encoder(ego_info)
        action_enc = self.action_encoder(action)
        state_ = torch.cat((left_enc, center_enc, right_enc, ego_enc, action_enc), dim=1)
        hidden = F.relu(self.fc(state_))
        out = self.fc_out(hidden)
        return out


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
        