import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import carla
import weakref
import torch
import os, sys
from datetime import datetime
from collections import deque
from multiprocessing import Process,Queue
#from macad_gym import LOG_PATH
sys.path.append(os.getcwd())

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def info(title):
#     print(title)
#     print('module name:',__name__)
#     print('parent process:',os.getpid())
#     print('process id:',os.getpid())

# def f(name):
#     info('function f')
#     print('hello',name)

# if __name__=='__main__':
#     info('main line')
#     p=Process(target=f,args=('bob',))
#     p.start()
#     p.join()

"""Exchanging objects between processes"""
def d(q):
    q.put([41,None,'hh'])
def f(q):
    q.put([42,None,'hello'])

# 平滑处理，类似tensorboard的smoothing函数。
def smooth(read_path, save_path, file_name, x='Step', y='Value', weight=0.99):
    data = pd.read_csv(read_path + file_name)
    scalar = data[y].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({x: data[x].values, y: smoothed})
    save.to_csv(save_path + 'smooth_'+ file_name)

class temp:
    def __init__(self) -> None:
        self.temp = 0

def tt(t):
    self = t()
    if self is None:
        return
    print("hhj")

if __name__=='__main__':
    # plt.style.use('ggplot')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # # 平滑预处理原始reward数据
    # smooth(read_path='./out/', save_path='./out/', file_name='reward.csv')
    # smooth(read_path='./out/', save_path='./out/', file_name='reward_1.csv')
    # smooth(read_path='./out/', save_path='./out/', file_name='reward_2.csv')
    # # 读取平滑后的数据
    # df1 = pd.read_csv('./out/smooth_reward.csv')  
    # df2 = pd.read_csv('./out/smooth_reward_1.csv')
    # df3 = pd.read_csv('./out/smooth_reward_2.csv') 
    # print(df1)
    # print(df2)
    # # 拼接到一起
    # df = df1.append(df2.append(df3))
    # # 重新排列索引
    # df.index = range(len(df))
    # print(df1)
    # # 设置图片大小
    # plt.figure(figsize=(15, 10))
    # # 画图
    # sns.lineplot(data=df, x="Step", y="Value")
    # plt.show()

    data = temp()
    #data = np.random.random([3,3])
    # 刚刚建立对象时的引用数
    print(sys.getrefcount(data))			# 此时引用数为 2
    # 建立弱引用
    ref = weakref.ref(data)
    tt(ref)
    # 查看增加弱引用后的引用数
    data.temp = 1
    print(data.temp)
    print(ref().temp)
    #del data
    print(sys.getrefcount(data))			# 此时引用数仍为 2，表明弱引用不增加引用数
    # 如果为弱引用对象增加强引用，引用数会增加
    print(ref())

    prox_ref = ref()
    del data
    print(prox_ref)
  
    prox_ref.temp =5
    print(sys.getrefcount(prox_ref))

    print(sys.getrefcount(prox_ref))
    print(ref().temp)
    #print(sys.getrefcount(data))			# 引用数为 3，不要为弱引用对象增加强引用
	# 二者输出 id 相同，表明弱引用对象指向同一内存空间
    print(id(ref()))						# 2809935694304
    #print(id(data))							# 2809935694304
    # 返回为 True 表明二者为同一对象
    print(ref() is data)					# True
    # 对象本身为弱引用对象
    print(ref)								# <weakref at 0x0000028E3D380A40; to 'numpy.ndarray' at 0x0000028E3D3809E0>
    # 类型为 弱引用
    print(type(ref))						# <class 'weakref'>
    # 引用对象时和原始内容一致
    print(type(ref()))						# <class 'numpy.ndarray'>
    # 数据内容完全一样
    print(ref())
    print(data)
