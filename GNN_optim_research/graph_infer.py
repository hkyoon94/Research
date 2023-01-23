import os
clc = lambda: os.system('cls')
cwd = os.getcwd()
desktop = "C:/Users/echo_/Desktop/"

import torch_utils as utils
from torch_utils import pipeline, task, monitor, datatool, auxs
import torch as t
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P
import numpy as np
import matplotlib.pyplot as plt
from math import nan

D = datatool.my_data_generator(p_train=0.4, p_test=0.6, remainder_twist=True)
D.load_cuda()


#--------------------------------------------------------------------------------------------------#
# A leaning idea 1
  
class NodeClassifier(t.nn.Module):
    def __init__(self, seed, D, hidden_channels, A):
        super().__init__()
        self.D = D
        
        self.A = P(A)
        t.manual_seed(seed); 
        self.W0 = P(t.randn(hidden_channels,D.num_features))
        self.b0 = P(t.randn(hidden_channels,1))
        self.W1 = P(t.randn(D.num_classes,hidden_channels))
        self.b1 = P(t.randn(D.num_classes,1))
        self.params = list(self.parameters())
        self.n_params = len(self.params)

    def forward(self, x):
        x = t.tensordot(x,self.A,dims=[[1],[0]])
        x = t.tensordot(self.W0,x,dims=[[1],[0]])+self.b0
        #x = F.dropout(x,p=0.5,training=self.training)
        x = t.sigmoid(x)
        x = t.tensordot(x,self.A,dims=[[1],[0]])
        x = t.tensordot(self.W1,x,dims=[[1],[0]])+self.b1
        return t.transpose(x,0,1)
  
  
class NodeClassifier_test(t.nn.Module):
    def __init__(self, A):
        super().__init__()
        
        self.A = A
        self.W0 = node_train_model.W0
        self.b0 = node_train_model.b0
        self.W1 = node_train_model.W1
        self.b1 = node_train_model.b1
        self.params = [self.A,self.W0,self.b0,self.W1,self.b1]

    def forward(self, x):
        x = t.tensordot(x,self.A,dims=[[1],[0]])
        x = t.tensordot(self.W0,x,dims=[[1],[0]])+self.b0
        x = F.sigmoid(x)
        x = t.tensordot(x,self.A,dims=[[1],[0]])
        x = t.tensordot(self.W1,x,dims=[[1],[0]])+self.b1
        return t.transpose(x,0,1)
  

seed = 1; t.manual_seed(seed)
Adj = 0.01*t.randn(D.num_train,D.num_train,device=0)

node_train_model =  NodeClassifier(seed,D,hidden_channels=8,A=Adj).cuda()

result = pipeline.BasicLearn(
    model = node_train_model,
    D = D,
    criterion = t.nn.CrossEntropyLoss(),
    optimizer = t.optim.Adam(node_train_model.parameters(), lr=5e-3, weight_decay=1e-5),
    total_epoch = 1200,
    train_x = t.transpose(D.x[D.train_mask],0,1), 
    train_y = D.y[D.train_mask],
    tasklist = [ 
                (task.Prog.loss_basic, 1), # int: per | nan: only at start & end
                (task.Prog.acc_maskwise, 1),
                (monitor.Prog.loss_log, 20),
                (monitor.Prog.acc_log, 20),
                (monitor.Prog.LA_plot, 100),
                (task.Param.P_max_mean, 20),
                (monitor.Param.P_max_mean_log, 100),
                (monitor.Param.P_max_mean_window, 100),
                (monitor.X.out_2d, nan),
                (monitor.Param.P_show, nan),
              ],
    save_dir = f"{cwd}/results_node_train_0_cuda",
    )

print(f"Best fitting Adj: {node_train_model.A}")
utils.P_draw(node_train_model.A,size=(15,7)); plt.show()
   
   
    
#--------------------------------------------------------------------------------------------------#
# extrapolation test

trained_A = node_train_model.A.detach().numpy()

ind = D.num_classes*[None]
for i in range(0,D.num_classes):
    ind[i] = D.y[D.train_mask] == i

trained_A_cluster = np.zeros((D.num_classes, D.num_classes))

for i in range(0,D.num_classes):
    for j in range(0,D.num_classes):
        trained_A_cluster[i,j] =  trained_A[ind[i]][:,ind[j]].mean()
    
utils.P_draw([node_train_model.A, t.tensor(trained_A_cluster)]).show()

extrapolated_A = np.zeros((D.num_test,D.num_test))

Ind = D.num_classes*[None]
for i in range(0,D.num_classes):
    Ind[i] = D.y[D.test_mask] == i

for i in range(0,D.num_test):
    for j in range(0,D.num_test):
        nested_cond = False
        for k in range(0,D.num_classes):
            if nested_cond is True:
                break
            for l in range(0,D.num_classes):
                if Ind[k][i].item() is True and Ind[l][j].item() is True:
                    nested_cond = True
                    extrapolated_A[i,j] = trained_A_cluster[k,l]
                    break
        
extrapolated_A = t.tensor(extrapolated_A).float()
utils.P_draw(extrapolated_A); plt.show()

extrapolated_A_noise = t.tensor(extrapolated_A\
                                +0.2*np.random.randn(D.num_test,D.num_test)).float()
utils.P_draw(extrapolated_A_noise); plt.show()

  

print("Extrapolated A test set result:")
pipeline.BasicTest(model = NodeClassifier_test(A=extrapolated_A),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = t.transpose(D.x[D.test_mask],0,1), y = D.y[D.test_mask])

print("Extrapolated A_noise test set result:")
pipeline.BasicTest(model = NodeClassifier_test(A=extrapolated_A_noise),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = t.transpose(D.x[D.test_mask],0,1), y = D.y[D.test_mask])



#--------------------------------------------------------------------------------------------------#
# Trinary(or binary)-Thresholded A test
trained_A = node_train_model.A
trinary_A = t.zeros(D.num_train,D.num_train)

for i in range(0,D.num_train):
    for j in range(0,D.num_train):
        if trained_A[i,j].item() > 0.01:
            trinary_A[i,j] = 1
        elif trained_A[i,j].item() < 0.01:
            trinary_A[i,j] = -1
        else:
            trinary_A[i,j] = 0
  
pipeline.BasicTest(model = NodeClassifier_test(A = trinary_A),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = t.transpose(D.x[D.train_mask],0,1), y = D.y[D.train_mask])



#--------------------------------------------------------------------------------------------------#
# edge-class prediction

ind = D.num_classes*[None]
for i in range(0,D.num_classes):
    ind[i] = D.y[D.train_mask] == i

edge_x = t.zeros(2*D.num_features,D.num_train,D.num_train)
edge_y = t.zeros(D.num_train,D.num_train,dtype=t.long)

for i in range(0,D.num_train):
    for j in range(0,D.num_train):
        edge_x[:,i,j] = t.cat( (D.x[D.train_mask][i,:],D.x[D.train_mask][j,:]), dim=0 )
        nested_cond=False
        for k in range(0,D.num_classes):
            if nested_cond is True:
                break
            for l in range(0,D.num_classes):
                if ind[k][i].item() is True and ind[l][j].item() is True:
                    edge_y[i,j] = 4*k+l
                    nested_cond = True
                    break
            
  
class EdgePredictor(t.nn.Module):
    def __init__(self,seed,edge_x,hidden_channels):
        super().__init__()
        t.manual_seed(seed); 

        self.W0 = P(t.randn(hidden_channels[0],edge_x.shape[0]))
        self.b0 = P(t.randn(hidden_channels[0],1,1))
        self.W1 = P(t.randn(hidden_channels[1],hidden_channels[0]))
        self.b1 = P(t.randn(hidden_channels[1],1,1))
        self.W2 = P(t.randn(1,hidden_channels[1]))
        self.b2 = P(t.randn(1,1))
        self.params = list(self.parameters())
        self.n_params = len(self.params)

    def forward(self, x):
        x = t.tensordot(self.W0,x,dims=[[1],[0]])+self.b0
        #x = F.dropout(x,p=0.5,training=self.training)
        x = t.sigmoid(x)
        x = t.tensordot(self.W1,x,dims=[[1],[0]])+self.b1
        #x = F.dropout(x,p=0.5,training=self.training)
        x = t.sigmoid(x)
        x = t.tensordot(self.W2,x,dims=[[1],[0]])+self.b2
        return x


# experiment 1
edge_train_model = EdgePredictor(seed, edge_x, hidden_channels=[200,100] )
criterion = t.nn.MSELoss()
optimizer = t.optim.Adam(edge_train_model.parameters(), lr=5e-3, weight_decay=1e-5)
total_epoch = 100000
train_x = edge_x; train_y = edge_y
val_x=[]; val_y=[]; test_x=[]; test_y=[]
tasklist = [ 
              (task.Prog.loss_basic, 1), # int: per | nan: only at end
              (monitor.Prog.loss_log, 20),
              (monitor.Prog.LA_plot, 200),
              (task.Param.P_max_mean, 50),
              (monitor.Param.P_max_mean_log, 1e10),
              (monitor.Param.P_max_mean_window, 200),
              (monitor.Param.P_show, 1e10),
            ]
streaming = True 
#save_dir = f"{cwd}/results_edge_train0"
result = pipeline.BasicLearn(edge_train_model,criterion,optimizer,total_epoch,train_x,train_y,
                             tasklist,val_x,val_y,test_x,test_y,streaming)



#--------------------------------------------------------------------------------------------------#
# processing for strictly-partitioned regression edge learning

etm, evm, _, = \
  datatool.masker(D.num_train,p_train=0.7,p_val=0.3,p_test=0)

etm.cuda(); evm.cuda()
edge_y_tr = t.zeros(etm.sum(),etm.sum(), dtype=t.float, device=0)
edge_x_tr = t.zeros(etm.sum(),etm.sum(), 2*D.num_features, device=0)
edge_y_val = t.zeros(evm.sum(),evm.sum(), dtype=t.float, device=0)
edge_x_val = t.zeros(evm.sum(),evm.sum(), 2*D.num_features, device=0)
#D.unload_cuda()

target_A = 100*node_train_model.A

for i in range(0,etm.sum()):
    for j in range(0,etm.sum()):
        edge_y_tr[i,j] = target_A[etm][:,etm][i,j].item()
        edge_x_tr[i,j,:] = t.cat( (D.x[D.train_mask][etm][i,:],D.x[D.train_mask][etm][j,:]), dim=0 )

for i in range(0,evm.sum()):
    for j in range(0,evm.sum()):
        edge_y_val[i,j] = target_A[evm][:,evm][i,j].item()
        edge_x_val[i,j,:] = t.cat( (D.x[D.train_mask][evm][i,:],D.x[D.train_mask][evm][j,:]), dim=0 )

edge_x_tr = edge_x_tr.view(-1,2*D.num_features)
edge_y_tr = edge_y_tr.view(-1,)
edge_x_val = edge_x_val.view(-1,2*D.num_features)
edge_y_val = edge_y_val.view(-1,)


# reg_y = t.zeros(D.num_train,D.num_train,device=0)
# reg_x = t.zeros(D.num_train,D.num_train,2*D.num_features,device=0)
# #D.unload_cuda()
# A = node_train_model.A

# for i in range(0,D.num_train):
#   for j in range(0,D.num_train):
#     reg_y[i,j] = 100*A[i,j].item()
#     reg_x[i,j,:] = t.cat( (D.x[D.train_mask][i,:],D.x[D.train_mask][j,:]), dim=0 )
    


#--------------------------------------------------------------------------------------------------#
# performing edge learning 
  
class WeightPredictor(t.nn.Module):
    def __init__(self,seed,reg_x,hidden_channels):
        super().__init__()
        t.manual_seed(seed); 

        self.W0 = P(t.randn(reg_x.shape[1],hidden_channels[0]))
        self.b0 = P(t.randn(hidden_channels[0]))
        self.W1 = P(t.randn(hidden_channels[0],hidden_channels[1]))
        self.b1 = P(t.randn(hidden_channels[1]))
        self.W2 = P(t.randn(hidden_channels[1],1))
        self.b2 = P(t.randn(1))
        self.params = list(self.parameters())
        self.n_params = len(self.params)

    def forward(self, x):
        x = t.tensordot(x,self.W0,dims=[[1],[0]])+self.b0
        #x = F.dropout(x,p=0.5,training=self.training)
        x = t.sigmoid(x)
        x = t.tensordot(x,self.W1,dims=[[1],[0]])+self.b1
        #x = F.dropout(x,p=0.5,training=self.training)
        x = t.sigmoid(x)
        x = t.tensordot(x,self.W2,dims=[[1],[0]])+self.b2
        return x


# experiment 1
edge_train_model = WeightPredictor(seed, edge_x_tr, hidden_channels=[200,100] ).cuda()
criterion = t.nn.MSELoss()
optimizer = t.optim.Adam(edge_train_model.parameters(), lr=5e-3, weight_decay=1e-5)
total_epoch = 100000
train_x = edge_x_tr; train_y = edge_y_tr
val_x=edge_x_val; val_y=edge_y_val; test_x=[]; test_y=[]
tasklist = [ 
              (task.Prog.loss_basic, 1), # int: per | nan: only at end
              (monitor.Prog.loss_log, 20),
              (monitor.Prog.LA_plot, 500),
              (task.Param.P_max_mean, 50),
              #(monitor.Param.P_max_mean_log, 1e10),
              (monitor.Param.P_max_mean_window, 500),
              (monitor.Param.P_show, 1e10),
            ]
streaming = True
#save_dir = f"{cwd}/results_edge_train0"
save_dir = False
result = pipeline.BasicLearn(edge_train_model,D,criterion,optimizer,total_epoch,train_x,train_y,
                             tasklist,val_x,val_y,test_x,test_y,streaming,save_dir)
