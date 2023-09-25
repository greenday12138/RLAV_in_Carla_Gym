import os
import random, collections
import numpy as np
import torch,logging
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from algs.util.replay_buffer import SumTree,SplitReplayBuffer
from macad_gym.viz.logger import LOG



class PriReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    Prioritized experience replay
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    Detailed information:
    https://yulizi123.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority. If alpha = 0, there is no Importance Sampling.
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        #assert self.tree.size==self.tree.capacity
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        b_state,b_action,b_action_param,b_reward,b_next_state,b_truncated,b_done,b_info=[],[],[],[],[],[],[],[]
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[self.tree.capacity-1:self.tree.capacity-1+self.size()]) / self.tree.total_p
        #min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b) #sample from  [a, b) 
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i]=idx
            b_state.append(data[0])
            b_action.append(data[1])
            b_action_param.append(data[2])
            b_reward.append(data[3])
            b_next_state.append(data[4])
            b_truncated.append(data[5])
            b_done.append(data[6])
            b_info.append(data[7])

        # print(self.tree.tree)
        # print(b_idx)
        return b_idx, ISWeights, (b_state,b_action,b_action_param,b_reward,b_next_state,b_truncated,b_done,b_info)

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
    
    def size(self):
        return self.tree.size


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
        self.ego_encoder = nn.Linear(state_dim['hero_vehicle'], self.hidden_size)
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
        super().__init__()
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.action_parameter_size = action_parameter_size
        self.train = train
        self.left_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.center_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.right_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.ego_encoder = nn.Linear(self.state_dim['hero_vehicle'], 64)
        self.fc = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, self.action_parameter_size)
        self.fc_std = nn.Linear(256, self.action_parameter_size)

    def forward(self, state):
        # state: (waypoints + 2 * companion_vehicle * 3
        one_state_dim = self.state_dim['waypoints'] + self.state_dim['companion_vehicle'] * 2 + self.state_dim['light']
        ego_info = state[:, 3*one_state_dim:]

        left_enc = self.left_encoder(state[:, :one_state_dim], ego_info)
        center_enc = self.center_encoder(state[:, one_state_dim:2*one_state_dim], ego_info)
        right_enc = self.right_encoder(state[:, 2*one_state_dim:3*one_state_dim], ego_info)
        ego_enc = self.ego_encoder(ego_info)
        state_ = torch.cat((left_enc, center_enc, right_enc, ego_enc), dim=1)
        
        hidden = F.relu(self.fc(state_))
        mu = self.fc_mu(hidden)
        log_std = self.fc_std(hidden).tanh()
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = dist.log_prob(normal_sample) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

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
        self.ego_encoder = nn.Linear(self.state_dim['hero_vehicle'], 64)
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
        self.ego_encoder = nn.Linear(self.state_dim['hero_vehicle'], 32)
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


class QValueNet_multi_td3(torch.nn.Module):
    def __init__(self, state_dim, action_param_dim, num_actions) -> None:
        # parameter state_dim here is a dict
        super().__init__()
        self.state_dim = state_dim
        self.action_param_dim = action_param_dim
        self.num_actions = num_actions
        self.left_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.center_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.right_encoder = lane_wise_cross_attention_encoder(self.state_dim)
        self.ego_encoder = nn.Linear(self.state_dim['hero_vehicle'], 32)
        self.action_encoder = nn.Linear(self.action_param_dim, 32)
        self.fc = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, self.num_actions)

        self.left_encoder2 = lane_wise_cross_attention_encoder(self.state_dim)
        self.center_encoder2 = lane_wise_cross_attention_encoder(self.state_dim)
        self.right_encoder2 = lane_wise_cross_attention_encoder(self.state_dim)
        self.ego_encoder2 = nn.Linear(self.state_dim['hero_vehicle'], 32)
        self.action_encoder2 = nn.Linear(self.action_param_dim, 32)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out2 = nn.Linear(256, self.num_actions)

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

        left_enc2 = self.left_encoder2(state[:, :one_state_dim], ego_info)
        center_enc2 = self.center_encoder2(state[:, one_state_dim:2*one_state_dim], ego_info)
        right_enc2 = self.right_encoder2(state[:, 2*one_state_dim:3*one_state_dim], ego_info)
        ego_enc2 = self.ego_encoder2(ego_info)
        action_enc2 = self.action_encoder2(action)
        state_2 = torch.cat((left_enc2, center_enc2, right_enc2, ego_enc2, action_enc2), dim=1)
        hidden2 = F.relu(self.fc(state_2))
        out2 = self.fc_out(hidden2)
        return out, out2


class P_SAC:
    def __init__(self, state_dim, action_dim, action_bound, gamma, tau,
                 buffer_size, batch_size, actor_lr, critic_lr, alpha_lr,
                 clip_grad, zero_index_gradients, inverting_gradients, per_flag, device) -> None:
        self.learn_time = 0
        self.replace_a = 0
        self.replace_c = 0
        self.s_dim = state_dim  # state_dim here is a dict
        self.s_dim['waypoints'] *= 3  # 2 is the feature dim of each waypoint
        self.a_dim, self.a_bound = action_dim, action_bound
        self.num_actions = 3  # left change, lane follow, right change
        self.action_parameter_sizes = np.array([self.a_dim, self.a_dim, self.a_dim])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)  # [0, self.a_dim, self.a_dim*2, self.a_dim*3]
        self.action_parameter_max_numpy = np.array([1, 1, 1, 1, 1, 1])
        self.action_parameter_min_numpy = np.array([-1, -1, -1, -1, -1, -1])
        self.action_parameter_range_numpy = np.array([2, 2, 2, 2, 2, 2])
        self.gamma, self.tau = gamma, tau 
        self.batch_size, self.device = batch_size, device
        self.actor_lr, self.critic_lr, self.alpha_lr = actor_lr, critic_lr, alpha_lr
        self.clip_grad = clip_grad
        self.indexd = zero_index_gradients
        self.zero_index_gradients = zero_index_gradients
        self.inverting_gradients = inverting_gradients
        self.td3 = False
        self.policy_freq = 2
        self.per_flag=per_flag
        self.learn_time=0
        # adjust different types of replay buffer
        #self.replay_buffer = Split_ReplayBuffer(buffer_size)
        if not self.per_flag:
            self.replay_buffer = SplitReplayBuffer(buffer_size)
            self.loss = nn.MSELoss()
        else:
            self.replay_buffer = PriReplayBuffer(buffer_size)
            self.loss = nn.MSELoss(reduction='none')
        # self.replay_buffer = offline_replay_buffer()
        """self.memory=torch.tensor((buffer_size,self.s_dim*2+self.a_dim+1+1),
            dtype=torch.float32).to(self.device)"""
        self.pointer = 0  # serve as updating the memory data
        self.train = True
        self.actor = PolicyNet_multi(self.s_dim, self.action_parameter_size, self.a_bound).to(self.device)
        if not self.td3:
            self.critic1 = QValueNet_multi(self.s_dim, self.action_parameter_size, self.num_actions).to(self.device)
            self.critic2 = QValueNet_multi(self.s_dim, self.action_parameter_size, self.num_actions).to(self.device)
            self.critic1_target = QValueNet_multi(self.s_dim, self.action_parameter_size, self.num_actions).to(self.device)
            self.critic2_target = QValueNet_multi(self.s_dim, self.action_parameter_size, self.num_actions).to(self.device)
        else:
            self.critic1 = QValueNet_multi_td3(self.s_dim, self.action_parameter_size, self.num_actions).to(self.device)
            self.critic2 = QValueNet_multi_td3(self.s_dim, self.action_parameter_size, self.num_actions).to(self.device)
            self.critic1_target = QValueNet_multi_td3(self.s_dim, self.action_parameter_size, self.num_actions).to(self.device)
            self.critic2_target = QValueNet_multi_td3(self.s_dim, self.action_parameter_size, self.num_actions).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -np.prod((self.action_parameter_size,)).item() # heuristic

    def take_action(self, state, lane_id=-2, action_mask=False):
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
        state_light = torch.tensor(state['light'], dtype=torch.float32).view(1, -1).to(self.device)
        state_ev = torch.tensor(state['hero_vehicle'],dtype=torch.float32).view(1,-1).to(self.device)
        state_ = torch.cat((state_left_wps, state_veh_left_front, state_veh_left_rear, state_light,
                            state_center_wps, state_veh_front, state_veh_rear, state_light,
                            state_right_wps, state_veh_right_front, state_veh_right_rear, state_light, state_ev), dim=1)
        # print(state_.shape) 
        all_action_param, log_prob = self.actor(state_)
        if not self.td3:
            q1 = self.critic1(state_, all_action_param)
            q2 = self.critic2(state_, all_action_param)
            q = torch.min(q1,  q2)
        else:
            q1, q2 = self.critic(state_, all_action_param)
            q = torch.min(q1, q2)
        q_a = torch.squeeze(q)
        q_a = q_a.detach().cpu().numpy()
        if action_mask:
            if lane_id == -3:
                q_a[2] = -1000000.0
            elif lane_id == -1:
                q_a[0] = -1000000.0
        action = np.argmax(q_a)
        action_param = all_action_param[:, self.action_parameter_offsets[action]:self.action_parameter_offsets[action+1]]

        LOG.psac_logger.debug(f"Network Output - Action: {action}, Steer: {action_param[0][0]}, Throttle_brake: {action_param[0][1]}")
        LOG.psac_logger.debug(f"q values:{q_a}")
        if (action_param[0, 0].is_cuda):
            action_param = np.array([action_param[:, 0].detach().cpu().numpy(), action_param[:, 1].detach().cpu().numpy()]).reshape((-1, 2))
            all_action_param = np.array([all_action_param[:, 0].detach().cpu().numpy(), all_action_param[:, 1].detach().cpu().numpy(),
                                        all_action_param[:, 2].detach().cpu().numpy(), all_action_param[:, 3].detach().cpu().numpy(),
                                        all_action_param[:, 4].detach().cpu().numpy(), all_action_param[:, 5].detach().cpu().numpy()]).reshape((-1, 6))
        else:
            action_param = np.array([action_param[:, 0].detach().numpy(), action_param[:, 1].detach().numpy()]).reshape((-1, 2))
            all_action_param = np.array([all_action_param[:, 0].detach().numpy(), all_action_param[:, 1].detach().numpy(),
                                        all_action_param[:, 2].detach().numpy(), all_action_param[:, 3].detach().numpy(),
                                        all_action_param[:, 4].detach().numpy(), all_action_param[:, 5].detach().numpy()]).reshape((-1, 6))

        return action, action_param, all_action_param

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices
            # print('actual_index: ', actual_index)
            grad[actual_index] = 0.
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        if grad_type == "action_parameters":
            max_p = torch.from_numpy(self.action_parameter_max_numpy)
            min_p = torch.from_numpy(self.action_parameter_min_numpy)
            rnge = torch.from_numpy(self.action_parameter_range_numpy)
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def learn(self):
        self.learn_time += 1
        # if self.learn_time > 100000:
        #     self.train = False
        self.replace_a += 1
        self.replace_c += 1
        if not self.per_flag:
            b_s, b_a, b_a_param, b_r, b_ns, b_t, b_d = self.replay_buffer.sample(self.batch_size)
        else:
            b_idx,b_ISWeights,b_transition = self.replay_buffer.sample(self.batch_size)
            b_s, b_a, b_a_param, b_r, b_ns, b_t, b_d, b_i = b_transition[0], b_transition[1], b_transition[2], b_transition[3], b_transition[4], \
                b_transition[5], b_transition[6],b_transition[7]
            self.ISWeights=torch.tensor(b_ISWeights,dtype=torch.float32).view((self.batch_size,-1)).to(self.device)

        # 此处得到的batch是否是pytorch.tensor?
        batch_s = torch.tensor(np.array(b_s), dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_ns = torch.tensor(np.array(b_ns), dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_a = torch.tensor(np.array(b_a), dtype=torch.int64).view((self.batch_size, -1)).to(self.device)
        batch_a_param = torch.tensor(np.array(b_a_param), dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_r = torch.tensor(np.array(b_r), dtype=torch.float32).view((self.batch_size, -1)).to(self.device).squeeze()
        batch_d = torch.tensor(np.array(b_d), dtype=torch.float32).view((self.batch_size, -1)).to(self.device).squeeze()
        batch_t = torch.tensor(np.array(b_t), dtype=torch.float32).view((self.batch_size, -1)).to(self.device).squeeze()

        with torch.no_grad():
            action_param_target, log_prob = self.actor(batch_ns)
            entropy = -log_prob
            if not self.td3:
                q_target_values1 = self.critic1_target(batch_ns, action_param_target)
                q_target_values2 = self.critic2_target(batch_ns, action_param)
                q_target_values = torch.min(q_target_values1, q_target_values2) + self.log_alpha.exp() * entropy
            else:
                q_target_values1, q_target_values2 = self.critic_target(batch_ns, action_param_target)
                q_target_values = torch.min(q_target_values1, q_target_values2)
            q_prime = torch.max(q_target_values, 1, keepdim=True)[0].squeeze()
            q_targets = batch_r + self.gamma * q_prime * (1 - batch_t) * (1 - batch_d)
        if not self.td3:
            q1_values = self.critic1(batch_s, batch_a_param)
            q2_values = self.critic2(batch_a, batch_a_param)
            q1 = q1_values.gather(1, batch_a.view(-1, 1)).squeeze()
            q2 = q2_values.gather(1, batch_a.view(-1, 1)).squeeze()
            if not self.per_flag:
                loss_q = self.loss(q, q_targets)
            else:
                abs_loss = torch.abs(torch.min(q1, q2) - q_targets)
                abs_loss = np.array(abs_loss.detach().cpu().numpy())
                self.replay_buffer.batch_update(b_idx, abs_loss)

                critic_1_loss = torch.mean(self.loss(q1, q_targets) * self.ISWeights)
                critic_2_loss = torch.mean(self.loss(q2, q_targets) * self.ISWeights)
        else:
            q_values1, q_values2 = self.critic(batch_s, batch_a_param)
            q_values = torch.min(q_values1, q_values2)
            q = q_values.gather(1, batch_a.view(-1, 1)).squeeze()
            loss_q = self.loss(q, q_values1) + self.loss(q, q_values2)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip_grad)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.learn_time % self.policy_freq == 0:
            with torch.no_grad():
                action_param = self.actor(batch_s)
            action_param.requires_grad = True
            if not self.td3:
                Q = self.critic(batch_s, action_param)
                Q_val = Q
            else:
                Q1, Q2 = self.critic(batch_s, action_param)
                Q_val = torch.min(Q1, Q2)
            if self.indexd:
                Q_indexed = Q_val.gather(1, batch_a.view(-1, 1))
                Q_loss = torch.mean(Q_indexed)
            else:
                Q_loss = torch.mean(torch.sum(Q_val, 1))

            self.critic.zero_grad()
            Q_loss.backward()
            from copy import deepcopy
            # print('check batch_s whether has grad: ', batch_s.grad_fn)
            delta_a = deepcopy(action_param.grad.data)

            action_param = self.actor(Variable(batch_s))
            delta_a[:] = self._invert_gradients(delta_a, action_param, grad_type="action_parameters", inplace=True)
            if self.zero_index_gradients:
                delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=batch_a, inplace=True)

            out = -torch.mul(delta_a, action_param)
            self.actor.zero_grad()
            out.backward(torch.ones(out.shape).to(self.device))
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
            self.actor_optimizer.step()

        # update alpha value
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return np.mean([critic_1_loss.detach().cpu().numpy(), critic_2_loss.detach().cpu().numpy()])

    def _print_grad(self, model):
        '''Print the grad of each layer'''
        for name, parms in model.named_parameters():
            LOG.psac_logger.debug(f"-->name:{name}, -->grad_requires:{parms.requires_grad}, -->grad_value:{parms.grad}")

    def set_sigma(self, sigma_steer, sigma_acc):
        # self.sigma = sigma
        self.steer_noise = sigma_steer
        self.tb_noise = sigma_acc

    def reset_noise(self):
        pass
        # self.steer_noise.reset()
        # self.tb_noise.reset()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, net, target_net):
        net.load_state_dict(target_net.state_dict())

    def store_transition(self, state, action, action_param, reward, next_state, truncated, done, info):  
        # how to store the episodic data to buffer
        def _compress(state):
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
            state_ev = np.array(state['hero_vehicle'], dtype=np.float32).reshape((1, -1))
            state_light = np.array(state['light'], dtype=np.float32).reshape((1, -1))
            state_ = np.concatenate((state_left_wps, state_veh_left_front, state_veh_left_rear, state_light,
                                        state_center_wps, state_veh_front, state_veh_rear, state_light,
                                        state_right_wps, state_veh_right_front, state_veh_right_rear, state_light, state_ev), axis=1)
            return state_

        state=_compress(state)
        next_state=_compress(next_state)

        # if reward_ttc < -0.1 or reward_eff < 3:
        #     self.change_buffer.append((state, action, action_param, reward, next_state, truncated, done))
        # if truncated:
        #     self.change_buffer.append((state, action, action_param, reward, next_state, truncated, done))
        if not self.per_flag:
            if action == 0 or action == 2:
                self.replay_buffer.add((state, action, action_param, reward, next_state, truncated, done),False)
            self.replay_buffer.add((state, action, action_param, reward, next_state, truncated, done),True)
        else:
            self.replay_buffer.add((state, action, action_param, reward, next_state, truncated, done,info))
        # print("their shapes", state, action, next_state, reward_list, truncated, done)
        # state: [1, 28], action: [1, 2], next_state: [1, 28], reward_list = [1, 6], truncated = [1, 1], done = [1, 1]
        # all: [1, 66]

        return

    def save_net(self,file = None):
        state = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict()
        }
        torch.save(state, file)

    def load_net(self, file = None, map_location = torch.device('cpu')):
        if file is not None:
            state = torch.load(file, map_location=map_location)
            if 'actor' in state:
                self.actor.load_state_dict(state['actor'])
            if 'actor_optimizer' in state:
                self.actor_optimizer.load_state_dict(state['actor_optimizer'])
            if 'critic1' in state:
                self.critic1.load_state_dict(state['critic1'])
            if 'critic2' in state:
                self.critic2.load_state_dict(state['critic2'])
            if 'critic1_target' in state:
                self.critic1_target.load_state_dict(state['critic1_target'])
            if 'critic2_target' in state:
                self.critic2_target.load_state_dict(state['critic2_target'])
            if 'critic1_optimizer' in state:
                self.critic1_optimizer.load_state_dict(state['critic1_optimizer'])
            if 'critic2_optimizer' in state:
                self.critic2_target.load_state_dict(state['critic2_optimizer'])