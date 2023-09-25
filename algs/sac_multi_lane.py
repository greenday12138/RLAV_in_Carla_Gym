import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from algs.util.replay_buffer import ReplayBuffer, PriReplayBuffer


class lane_wise_cross_attention_encoder(torch.nn.Module):
    def __init__(self, state_dim, hidden_size, train=True):
        super().__init__()
        self.state_dim = state_dim
        self.train = train
        self.hidden_size = hidden_size
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


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_param_dim, action_bound, train=True):
        # the action bound and state_dim here are dicts
        super(PolicyNetContinuous, self).__init__()
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.action_param_dim = action_param_dim
        self.train = train
        self.left_encoder = lane_wise_cross_attention_encoder(self.state_dim, 64)
        self.center_encoder = lane_wise_cross_attention_encoder(self.state_dim, 64)
        self.right_encoder = lane_wise_cross_attention_encoder(self.state_dim, 64)
        self.ego_encoder = nn.Linear(self.state_dim['hero_vehicle'], 64)
        self.fc1 = torch.nn.Linear(256, 256)
        self.fc_mu = torch.nn.Linear(256, self.action_param_dim)
        self.fc_std = torch.nn.Linear(256, self.action_param_dim)

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

        x = F.relu(self.fc1(state_))
        mu = self.fc_mu(x)

        log_std = self.fc_std(x).tanh()
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)
    
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = dist.log_prob(normal_sample) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_param_dim):
        # parameter state_dim here is a dict
        super(QValueNetContinuous, self).__init__()
        self.state_dim = state_dim
        self.action_param_dim = action_param_dim
        self.left_encoder = lane_wise_cross_attention_encoder(self.state_dim, 64)
        self.center_encoder = lane_wise_cross_attention_encoder(self.state_dim, 64)
        self.right_encoder = lane_wise_cross_attention_encoder(self.state_dim, 64)
        self.ego_encoder = nn.Linear(self.state_dim['hero_vehicle'], 32)
        self.action_encoder = nn.Linear(self.action_param_dim, 32)
        self.fc = nn.Linear(256, 256)
        self.fc_out = torch.nn.Linear(256, 1)

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
        return self.fc_out(hidden)
    

class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, action_dim, action_bound, gamma, tau, 
                 buffer_size, batch_size, alpha_lr,
                 actor_lr, critic_lr, per_flag, device):
        self.learn_time = 0
        self.replace_a = 0
        self.replace_c = 0
        self.s_dim = state_dim  # state_dim here is a dict
        self.s_dim['waypoints'] *= 3  # 2 is the feature dim of each waypoint
        self.a_dim, self.a_bound = action_dim, action_bound
        self.gamma, self.tau = gamma, tau
        self.batch_size, self.device = batch_size, device
        self.actor_lr, self.critic_lr, self.alpha_lr = actor_lr, critic_lr, alpha_lr
        self.per_flag=per_flag
        self.train = True
        # adjust different types of replay buffer
        if self.per_flag:
            self.replay_buffer = PriReplayBuffer(buffer_size)
            self.loss = nn.MSELoss(reduction='none')
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
            self.loss = nn.MSELoss()
        self.actor = PolicyNetContinuous(self.s_dim, self.a_dim, self.a_bound).to(device)
        self.critic_1 = QValueNetContinuous(self.s_dim, self.a_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(self.s_dim, self.a_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(self.s_dim, self.a_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(self.s_dim, self.a_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -np.prod((action_dim,)).item() # heuristic

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
        state_light = torch.tensor(state['light'], dtype=torch.float32).view(1, -1).to(self.device)
        state_ev = torch.tensor(state['hero_vehicle'],dtype=torch.float32).view(1,-1).to(self.device)
        state_ = torch.cat((state_left_wps, state_veh_left_front, state_veh_left_rear, state_light,
                            state_center_wps, state_veh_front, state_veh_rear, state_light,
                            state_right_wps, state_veh_right_front, state_veh_right_rear, state_light, state_ev), dim=1)
        # print(state_.shape)
        action, log_prob = self.actor(state_)
        if (action[0, 0].is_cuda):
            action = np.array([action[:, 0].detach().cpu().numpy(), action[:, 1].detach().cpu().numpy()]).reshape((-1, 2))
        else:
            action = np.array([action[:, 0].detach().numpy(), action[:, 1].detach().numpy()]).reshape((-1, 2))
        return  action

    def calc_target(self, rewards, next_states, dones, truncateds):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones) * (1 - truncateds)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def learn(self):
        self.learn_time += 1
        self.replace_a += 1
        self.replace_c += 1

        if not self.per_flag:
            b_s, b_a, b_r, b_ns, b_t, b_d, b_i = self.replay_buffer.sample(self.batch_size)
        else:
            b_idx,b_ISWeights,b_transition = self.replay_buffer.sample(self.batch_size)
            b_s, b_a, b_r, b_ns, b_t, b_d, b_i = b_transition[0], b_transition[1], b_transition[2], b_transition[3], b_transition[4], \
                b_transition[5], b_transition[6]
            self.ISWeights=torch.tensor(b_ISWeights,dtype=torch.float32).view((self.batch_size,-1)).to(self.device)
            #print(self.ISWeights)

        batch_s = torch.tensor(b_s, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_ns = torch.tensor(b_ns, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_a = torch.tensor(b_a, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_r = torch.tensor(b_r, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_d = torch.tensor(b_d, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_t = torch.tensor(b_t, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        
        # update both Q network
        td_target = self.calc_target(batch_r, batch_ns, batch_d, batch_t)
        q1 = self.critic_1(batch_s, batch_a)
        q2 = self.critic_2(batch_s, batch_a)
        if not self.per_flag:
            critic_1_loss = torch.mean(self.loss(q1, td_target.detach()))
            critic_2_loss = torch.mean(self.loss(q2, td_target.detach()))
        else:
            abs_loss = torch.abs(torch.min(q1, q2) - td_target)
            abs_loss = np.array(abs_loss.detach().cpu().numpy())
            self.replay_buffer.batch_update(b_idx, abs_loss)

            critic_1_loss = torch.mean(self.loss(q1, td_target.detach()) * self.ISWeights)
            critic_2_loss = torch.mean(self.loss(q2, td_target.detach()) * self.ISWeights)
            
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # update policy network
        new_actions, log_prob = self.actor(batch_s)
        entropy = - log_prob
        q1_value = self.critic_1(batch_s, new_actions)
        q2_value = self.critic_2(batch_s, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update alpha value
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        loss_1 = critic_1_loss.detach().cpu().numpy()
        loss_2 = critic_2_loss.detach().cpu().numpy()
        return np.mean([loss_1, loss_2])

    def store_transition(self, state, action, reward, next_state, truncated, done, info):  # how to store the episodic data to buffer
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

        self.replay_buffer.add((state, action, reward, next_state, truncated, done,info))

        return
    def save_net(self,file = None):
        state = {
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.clone().detach(),
        }
        torch.save(state, file)

    def load_net(self, file = None, map_location = torch.device('cpu')):
        if file is not None:
            state = torch.load(file, map_location=map_location)
            if 'log_alpha' in state:
                self.log_alpha = state['log_alpha'].clone().detach().requires_grad_(True).to(map_location)
                self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
                # self.log_alpha = torch.tensor(state['log_alpha'], requires_grad=True, device=map_location)
            if 'actor' in state:
                self.actor.load_state_dict(state['actor'])
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            if 'critic_1' in state:
                self.critic_1.load_state_dict(state['critic_1'])
                self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
            if 'critic_2' in state:
                self.critic_2.load_state_dict(state['critic_2'])
                self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)
            if 'target_critic_1' in state:
                self.target_critic_1.load_state_dict(state['target_critic_1'])
            if 'target_critic_2' in state:
                self.target_critic_2.load_state_dict(state['target_critic_2'])
            # if 'actor_optimizer' in state:
            #     self.actor_optimizer.load_state_dict(state['actor_optimizer'])
            # if 'critic_1_optimizer' in state:
            #     self.critic_1_optimizer.load_state_dict(state['critic_1_optimizer'])
            # if 'critic_2_optimizer' in state:
            #     self.critic_2_optimizer.load_state_dict(state['critic_2_optimizer'])
            # if 'log_alpha_optimizer' in state:
            #     self.log_alpha_optimizer.load_state_dict(state['log_alpha_optimizer'])
           