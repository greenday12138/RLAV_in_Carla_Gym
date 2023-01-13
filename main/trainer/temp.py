import numpy as np
import random,logging
import carla
import torch
import datetime,os
from collections import deque
from multiprocessing import Process,Queue

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

if __name__=='__main__':
    # q=Queue()
    # p1=Process(target=f,args=(q,))
    # p2=Process(target=d,args=(q,))
    # p1.start()
    # p2.start()
    # print(q.get())
    # print(q.get())
    # p1.join()
    # p2.join()
    b=[2,3,4,5]
    a=[1,2,3]
    c=[]
    d=[]
    arr1=np.array(b,dtype=float)
    arr2=np.array(a,dtype=float)
    c.append(arr1)
    c.append(arr2)
    c=np.array(c)
    d.append(arr2)
    d.append(arr1)
    d=np.array(d)
    #print(c)
    np.save(f"./out/temp.npy", [c,d])
    c_ = np.load(f"./out/temp.npy",allow_pickle=True)
    #print(c_)
    
    temp=np.load(f"./out/rear_acc.npy",allow_pickle=True)
    for i in temp:
        print(i)
    