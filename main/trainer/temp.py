import numpy as np
import random,logging
import carla
import torch
import datetime,os
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
    q=Queue()
    p1=Process(target=f,args=(q,))
    p2=Process(target=d,args=(q,))
    p1.start()
    p2.start()
    print(q.get())
    print(q.get())
    p1.join()
    p2.join()