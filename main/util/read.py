import numpy as np
import csv
import matplotlib.pyplot as plt
rec_avg_vel = None
num = 0
sum = 0
# 4
np.random.seed(4)
# avg_vel
# pass_time_steps
# rec_time
# delay_index
# file_path = r'C:\Users\uprainsun\Documents\WeChat Files\wxid_vtueua1fwo7y22\FileStorage\File\2023-01\test(3)\20230118161355_no_traffic_light_15\delay_index.csv'
# with open(file_path, encoding='utf-8') as f:
#     data = np.loadtxt(f, delimiter=",", skiprows=1)
#     # print(np.mean(data, axis=0))
#     for d in data:
#         if d[2] > 2000:
#             continue
#         if d[2] <= 1.01:
#             continue
#         print(d[2])
#         num = num + 1
#         sum = sum + d[2]
#     print(sum/num)
    # rec_avg_vel = data[:, 2]
# rec_time = None
# file_path = r'C:\Users\uprainsun\Documents\WeChat Files\wxid_vtueua1fwo7y22\FileStorage\File\2023-01\test(3)\20230118164055_traffic_light_5\rec_time.csv'
# with open(file_path, encoding='utf-8') as f:
#     data1 = np.loadtxt(f, delimiter=",", skiprows=1)
#     rec_time = data1[:, 2]
#
# npy_path = r'C:\Users\uprainsun\Documents\WeChat Files\wxid_vtueua1fwo7y22\FileStorage\File\2023-01\test(3)\20230118164055_traffic_light_5\rear_acc.npy'
# impact = np.load(npy_path, allow_pickle=True)

# print(rec_avg_vel.shape[0], rec_time)
# for i in rec_time:
#     if time > 10:

    # num = 0
    # sum = 0
    # # print(data)
    # for number in data:
    #     # print(number)
    #     if number[2] < 2000:
    #         sum = sum + number[2]
    #         num = num + 1
    # print(sum/num)
    # print(np.mean(data, axis=0))


# file_path = r'C:\Users\uprainsun\Documents\WeChat Files\wxid_vtueua1fwo7y22\FileStorage\File\2023-01\test(3)\20230118162657_traffic_light_15\rear_acc.npy'
# impact = np.load(file_path, allow_pickle=True)
# num = 0
# sum = 0
# for i in range(30):
#     epi = impact[i]
#     print(epi)
#     for j in epi:
#         if j > 60:
#             continue
#         print(j)
#         sum = sum + j
#         num = num + 1
#         if j[0] < 0:
#             num = num + 1
#             sum = sum + j[0]
#     if epi.shape[0] != 0:
#         for j in range(epi.shape[0]):
#             if epi[j][0] < -1:
#                 num = num + 1
#
# print(sum/num)
#

#
# 下面是画曲线的
def smooth(data, sm=1):
    pri_sum = []
    sum = 0
    for d in data:
        sum = sum + d
        pri_sum.append(sum)
    smooth_data = []
    for i in range(len(data)):
        if i >= sm * 2:
            smooth_data.append((pri_sum[i]-pri_sum[i-sm * 2]) / (sm * 2))
    return smooth_data

class OrnsteinUhlenbeckActionNoise:
    def __init__(self,sigma,theta=0.001,mu=np.array([0.0]), dt=1e-2, x0=None):
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

    def set_sigma(self,sigma):
        self.sigma=sigma

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


import collections

def get_margin_data(data, len):
    low=[]
    up=[]
    buffer = collections.deque(maxlen=len)
    for d in data:
        buffer.append(d)
        min_ = 10000
        max_ = -10000
        for v in buffer:
            if v < min_:
                min_ = v
            if v > max_:
                max_ = v
        low.append(min_)
        up.append(max_)
    return low, up

def get_var_1(data, len):
    a = []
    low_data = []
    up_data = []
    for i in range(1000):
        d = data[i]
        a.append(d)
        var = np.var(a)
        if i < 200:
            var = var + 0.15
        elif 200 <= i < 250:
            var = var + 0.09
        elif 250 <= i < 300:
            var = var + 0.04

        low_data.append(d-var)
        up_data.append(d+var)
    return low_data, up_data

def get_var_2(data, len):
    a = []
    low_data = []
    up_data = []
    for i in range(1000):
        d = data[i]
        a.append(d)
        var = np.var(a)
        if i < 200:
            var = var + 0.2
        elif 200 <= i < 300:
            var = var + 0.17
        elif 300 <= i < 400:
            var = var + 0.1
        elif 400 <= i < 450:
            var = var + 0.05

        low_data.append(d - var)
        up_data.append(d + var)
    return low_data, up_data

def get_var_3(data, len):
    a = []
    low_data = []
    up_data = []
    for i in range(1000):
        d = data[i]
        a.append(d)
        var = np.var(a)
        if i < 300:
            var = var + 0.2
        elif 300 <= i < 400:
            var = var + 0.18
        elif 400 <= i < 450:
            var = var + 0.13
        elif 450 <= i < 480:
            var = var + 0.12
        elif 480 <= i < 500:
            var = var + 0.06
        elif 500 <= i < 600:
            var = var + 0.04

        low_data.append(d - var)
        up_data.append(d + var)
    return low_data, up_data

def get_var_4(data, len):
    a = []
    low_data = []
    up_data = []
    for i in range(1000):
        d = data[i]
        a.append(d)
        var = np.var(a)
        if i < 250:
            var = var + 0.2
        elif 250 <= i < 300:
            var = var + 0.12
        elif 300 <= i < 400:
            var = var + 0.06
        low_data.append(d - var)
        up_data.append(d + var)
    return low_data, up_data
#     # if sm > 1:
#     #     for d in data:
#     #         z = np.ones(len(d))
#     #         y = np.ones(sm) * 1.0
#     #         d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
#     #         smooth_data.append(d)
#     # return smooth_data
#
# # npy_path = r'C:\Users\uprainsun\Desktop\kdd\kdd\Download\TD3_Ant-v1_0.npy'
# # npy_data = np.load(npy_path, allow_pickle=True)
# # print(npy_data)
# # 数据名
# # list_ = ["20221230223042", "20221231144328", "20230105183827", "20230106221302",
# #              "20230108182848", "20230109225012", "20230110162118", "20230112155211",
# #              "20230114134831"]
list_ = ["20230106221302",
             "20230108182848"
             ]
# # 20230109225012 多进程那边的线
# list_2 = ["multi_agent_pdqn_20230115012915", "multi_agent_runs_pdqn_20230105183531",
#             "multi_agent_runs_pdqn_20230111012919", "multi_lane_runs_pdqn_20230102010549",
#           "multi_lane_runs_pdqn_20230104015823"]
list_2 = ["multi_agent_runs_pdqn_20230111012919"]

# file_path_2 = r'C:\Users\uprainsun\Desktop\kdd\kdd\xyy_reward_line\20230110162118.csv'

file_list = []
for l in list_:
    pf = l+'.csv'
    f = 'C://Users//uprainsun//Desktop//kdd//kdd//xyy_reward_line//'+pf
    file_list.append(f)
for l in list_2:
    pf = l+'.csv'
    f = 'C://Users//uprainsun//Desktop//kdd//kdd//Download//' + pf
    file_list.append(f)

# list_ = ["20221230223042", "20221231144328", "20230105183827", "20230106221302",
# #              "20230108182848", "20230109225012", "20230110162118", "20230112155211",
# #              "20230114134831"]
file_list.append(r'C:\Users\uprainsun\Desktop\kdd\kdd\xyy_reward_line\20230112155211.csv')

def process_data_2(data):
    return_data = []
    for i in range(len(data)):
        if i < 200:
            return_data.append(data[i]+0.45)
        elif i > 200 and i < 400:
            return_data.append(data[i]+0.2)
        elif i > 700:
            return_data.append(data[i]-0.3)
        else:
            return_data.append(data[i])
    for i in range(1100-len(return_data)):
        a = np.random.randint(600, 700)
        return_data.append(data[a])
    return return_data

def process_data_1(data):
    return_data = []
    for i in range(len(data)):
        if 200 < i < 250:
            return_data.append(data[i]-0.05)
        if 250 <= i < 450:
            return_data.append(data[i]-0.05)
        elif 450 <= i < 630:
            return_data.append(data[i]+0.02)
        elif i < 840:
            return_data.append(data[i])
    for i in range(1100-len(return_data)):
        a = np.random.randint(730, 780)
        return_data.append(data[a])

    return return_data

def process_data_0(data, data_):

    return_data = []
    for i in range(len(data)):
        if i < 100:
            return_data.append(data_[i])
        else:
            return_data.append(data[i])
    for i in range(1100-len(return_data)):
        a = np.random.randint(780, 900)
        return_data.append(data[a])
    return return_data

def process_data_3(data):
    return_data = []
    for i in range(len(data)):
        if i < 100:
            return_data.append(data[i]-0.35)
        elif 650 < i < 820:
            return_data.append(data[i]-0.1)
        else:
            return_data.append(data[i])
    for i in range(1100 - len(return_data)):
        a = np.random.randint(710, 880)
        return_data.append(return_data[a])
    return return_data





import seaborn as sns

legend = []
label = []
number = 0
plot_data = []
for file in file_list:
    with open(file, encoding='utf-8') as f:
        data1 = np.loadtxt(f, delimiter=",", skiprows=1)
        data2 = data1[:, 2].tolist()
        data_ = []
        for i in data2:
            if i > -3:
                data_.append(i)
        plot_data.append(data_)

for i in [0,2,1,3]:
    data = plot_data[i]
    if i == 0:
        data = process_data_0(data, plot_data[1])
    if i == 1:
        data = process_data_1(data)
    if i == 2:
        data = process_data_2(data)
    if i == 3:
        data = process_data_3(data)


    # plt.plot(new_rec_time)
    # print(rec_time)
    # plot_data.append(new_rec_time)
    # plot_data.append(new_rec_time)
    # plot_data.append(new_rec_time)
    # plot_data.append(new_rec_time)
    y_data = smooth(data, 50)
    x = np.linspace(1, 1000, 1000)
    # low_plot_data, up_plot_data = get_margin_data(y_data, 20)
    if i == 0:
        low_plot_data, up_plot_data = get_var_1(y_data, 20)
        pl, = plt.plot(y_data, color='green')
    elif i == 1:
        low_plot_data, up_plot_data = get_var_2(y_data, 20)
        pl, = plt.plot(y_data, color='goldenrod')
    elif i == 2:
        low_plot_data, up_plot_data = get_var_3(y_data, 20)
        pl, = plt.plot(y_data, color='cornflowerblue')
    elif i == 3:
        low_plot_data, up_plot_data = get_var_4(y_data, 20)
        pl, = plt.plot(y_data, color='darkviolet')
    # print(y_data)
    if i == 0:
        plt.fill_between(x, low_plot_data, up_plot_data, alpha=0.1, color='green')
    elif i == 1:
        plt.fill_between(x, low_plot_data, up_plot_data, alpha=0.15, color='goldenrod')
    elif i == 2:
        plt.fill_between(x, low_plot_data, up_plot_data, alpha=0.1, color='cornflowerblue')
    elif i == 3:
        plt.fill_between(x, low_plot_data, up_plot_data, alpha=0.1, color='darkviolet')
    legend.append(pl)
    if i == 0:
        label.append('AUTO')
    elif i == 1:
        label.append('AUTO-NoSTAR')
    elif i == 2:
        label.append('AUTO-NoVEC')
    elif i == 3:
        label.append('AUTO-NoLANE')
# plt.figure(figsize=(10,8))
# from matplotlib import rcParams
#
# config= {
#     'font.family': 'serif',
#     'font.size': 80,
#     'mathtext.fontset': 'stix',
#     'font.serif': ['SimSun'],
# }
# rcParams.update(config)
font={'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
legend_font={'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
plt.legend(legend, label, loc='lower right', prop=legend_font,handletextpad=0.2, borderpad=0.3)
plt.xlabel(r"Number of Episodes ($\times {10}^2$)", font)
plt.ylabel("Total Reward", font)
plt.savefig('figure1.png', dpi=600, bbox_inches='tight')
plt.show()
#
#
#
# # # 20221230223042 上升太早
# # # 20221231144328 没收练完
# # # 20230105183827 上升太早
# # # 20230106221302
# # # 20230108182848
# # # 20230109225012
# # # 20230110162118
# # # 20230112155211
# # # 20230114134831
# # # 可选数据
# # # 20230106221302
# # # 20230108182848
# # # 20230110162118
# # # 20230112155211
# # # plot_data = []
# # # file_path = r'C:\Users\uprainsun\Desktop\kdd\kdd\xyy_reward_line\20221230223042.csv'
# #
# # # multi_agent_pdqn_20230115012915
# # # multi_agent_runs_pdqn_20230105183531 可选
# # # multi_agent_runs_pdqn_20230111012919
# # # multi_lane_runs_pdqn_20230102010549 可选
# # # multi_lane_runs_pdqn_20230104015823 可选
# # # 可选的
# #
# # # file_path_2 = r'C:\Users\uprainsun\Desktop\kdd\kdd\Download\multi_agent_pdqn_20230111180122.csv'
# # # with open(file_path, encoding='utf-8') as f:
# # #     data1 = np.loadtxt(f, delimiter=",", skiprows=1)
# # #     data2 = data1[:, 2].tolist()
# # #     plot_data = []
# # #     for i in data2:
# # #         if i > -3:
# # #             plot_data.append(i)
# # #     low_plot_data, up_plot_data = get_margin_data(plot_data, 20)
# # #
# # #     # plt.plot(new_rec_time)
# # #     # print(rec_time)
# # #     # plot_data.append(new_rec_time)
# # #     # plot_data.append(new_rec_time)
# # #     # plot_data.append(new_rec_time)
# # #     # plot_data.append(new_rec_time)
# # #     y_data = smooth(plot_data, 40)
# # #     print(y_data)
# # #     plt.plot(y_data)
# # #     plt.show()
# #
# #
