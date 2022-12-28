import numpy as np
import random,logging
import carla
import torch
import datetime

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__=='__main__':
    # arr1=torch.tensor([[1],[-2]],dtype=torch.float32)
    # arr2=arr1.clone().detach()
    # for i in range(arr1.shape[0]):
    #     if arr1[i][0]<0:
    #         arr1[i][0]=0
    #     if arr2[i][0]>=0:
    #         arr2[i][0]=0
    # arr3=torch.cat((arr1,arr2),dim=1)
    # arr=np.array([[1,-1],[-1,-1]])
    # arr[:,0]+=arr[:,0]
    # sq=torch.tensor([[1]],dtype=torch.float32)
    # tt=torch.tensor(True,dtype=torch.float32)
    # tf=torch.tensor(False,dtype=torch.float32)
    #
    # # print(tt,tf,sep='\t')
    # # print(torch.squeeze(sq))
    # print(arr1,arr2,arr3,sep='\n')
    # print(torch.split(arr3,split_size_or_sections=[1,1],dim=1),sep='\n')
    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))