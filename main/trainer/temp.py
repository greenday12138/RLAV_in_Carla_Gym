import numpy as np
import random,logging
import carla
import torch
import datetime

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__=='__main__':
    li=[]
    arr=np.array([[1],np.array([1,2]),np.array([1,2,3])],dtype=object)
    print(arr,arr.shape,arr.size)
    