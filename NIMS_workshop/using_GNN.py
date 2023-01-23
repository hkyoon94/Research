import os
clc = lambda: os.system('cls')
cwd = os.getcwd()
desktop = "C:/Users/echo_/Desktop/"

import torch as t
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P
import torch_geometric as tg
import torch_utils as utils
from torch_utils import pipeline, task, monitor, datatool, nnframe
import numpy as np
import matplotlib.pyplot as plt
from math import nan
import dill
import pandas as pd
from itertools import combinations
  
  
#--------------------------------------------------------------------------------#

rawD = pd.read_excel(f"{cwd}/raw_data.xlsx",sheet_name='Pmod')
rawD.sort_values(by='Grade',ascending=True,inplace=True)
target = rawD["Grade"].values
x = rawD.iloc[:,2:-1].values

# MLP
A = t.eye(x.shape[0],x.shape[0])


#edge: composite cos simil
features = rawD[['FC','LTC','PC','OC','GCA','GCP']]
iter = list(combinations(features,4))

# for single-feature
# n=0
# acc_vec = np.zeros(len(features))
# for feature in iter:
#     v = rawD[feature[0]].values
    
#     A = t.zeros(v.shape[0],v.shape[0])
#     for i in range(v.shape[0]):
#         for j in range(i,v.shape[0]):
#             if abs(v[i]-v[j])<0.05:
#                 A[i,j]=1; A[j,i]=1

# for multi-feature
n=0
acc_vec = np.zeros(len(iter))

for feature in iter:
    v = rawD[list(feature)].values
    
    A = t.zeros(v.shape[0],v.shape[0])
    for i in range(v.shape[0]):
        for j in range(i,v.shape[0]):
            # if np.dot(v[i,:],v[j,:])/(np.linalg.norm(v[i,:],ord=2)*np.linalg.norm(v[j,:],ord=2)) > 0.999:
            if np.linalg.norm(v[i,:]-v[j,:],ord=2)<0.05:
                A[i,j]=1; A[j,i]=1
          

    #---------------------------------------------
    fig, ax = plt.subplots()
    ax.matshow(A)
    fig.suptitle(f"Adjacency by feature: {feature}")
    plt.savefig(f"{cwd}/{feature}_A.jpg")
    
    edge_index = A.to_sparse().indices()

    trm,vm,tm = datatool.masker(x.shape[0],p_train=0.20,p_val = 0,p_test=0.80)
    D = datatool.sortedDF(t.tensor(x).float(),t.tensor(target).long(),trm,vm,tm,edge_index)
    D.load_cuda()

    GNN = nnframe.FFNN(data = D, 
                      layers = [tg.nn.GCNConv, t.nn.Linear],
                      activations = [t.sigmoid, t.nn.Identity()],
                      hidden_channels = [4],
                      forward_opts = [[D.edge_index],[]],
                      dropouts=False, seed=False).cuda()

    tr_x=D.x[D.train_mask]; tr_y=D.y[D.train_mask]
    val_x=D.x[D.val_mask]; val_y=D.y[D.val_mask]
    test_x=D.x[D.test_mask]; test_y=D.y[D.test_mask]
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(GNN.parameters(),lr=0.005,weight_decay=1e-5)
    tasklist = [ 
                (task.Prog.loss_GNN, 1), # int: per | nan: only at start & end
                (task.Prog.acc_maskwise_GNN, 1),
                (monitor.Prog.loss_log, nan),
                (monitor.Prog.acc_log, nan),
                (monitor.Prog.LA_plot, nan),
                (monitor.X.out_2d_GNN, nan),
              ]
    streaming = False
    save_dir = f"{cwd}/{feature}"

    result = pipeline.BasicLearn(GNN, D, criterion, optimizer, 500,
                                tr_x,tr_y,val_x,val_y,test_x,test_y,tasklist, streaming, save_dir)
    result.start_learn()
    acc_vec[n] = result.acc_arr[2,:].max()
    n = n+1
    
print(acc_vec)
# for i in range(len(iter)):
#     iter[i] = ', '.join(iter[i])
# print(iter)

# plt.bar(iter,acc_vec)
# plt.grid(axis='y'); plt.ylim([0,100])
# plt.ylabel('Accuracy(%)')
# plt.xticks(rotation='vertical')
# plt.show()