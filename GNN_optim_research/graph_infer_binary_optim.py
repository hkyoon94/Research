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
import dill

D = datatool.my_data_generator(p_train=0.4, p_test=0.6, remainder_twist=True)
D.load_cuda()


#--------------------------------------------------------------------------------------------------#
# A learning idea 1

class NodeClassifier(t.nn.Module):
    def __init__(self, seed, D, hidden_channels, A):
        super().__init__()
        self.D = D
        self.A = P(A)
        t.manual_seed(seed); 
        self.W0 = P(t.randn(D.num_features,hidden_channels))
        self.b0 = P(t.randn(hidden_channels))
        self.W1 = P(t.randn(hidden_channels,D.num_classes))
        self.b1 = P(t.randn(D.num_classes))
        self.params = list(self.parameters())
        self.n_params = len(self.params)

    def forward(self, x):
        x = t.tensordot(t.sigmoid(100*t.matmul(t.transpose(self.A,0,1),self.A))\
            /D.num_train, x, dims=[[0],[0]])
        x = t.tensordot(x,self.W0,dims=[[1],[0]])+self.b0
        #x = F.dropout(x,p=0.5,training=self.training)
        x = t.sigmoid(x)
        x = t.tensordot(t.sigmoid(100*t.matmul(t.transpose(self.A,0,1),self.A))\
            /D.num_train, x, dims=[[0],[0]])
        x = t.tensordot(x,self.W1,dims=[[1],[0]])+self.b1
        return x
  
seed = 1; t.manual_seed(seed)
Adj = 0.01*t.randn(D.num_train,D.num_train,device=0)

node_train_model =  NodeClassifier(seed,D,hidden_channels=8,A=Adj).cuda()
result = pipeline.BasicLearn(
    model = node_train_model,
    D = D,
    criterion = t.nn.CrossEntropyLoss(),
    optimizer = t.optim.Adam(node_train_model.parameters(), lr=5e-3, weight_decay=1e-5),
    train_x = D.x[D.train_mask],
    train_y = D.y[D.train_mask],
    tasklist = [ 
                (task.Prog.loss_basic, 1), # int: per | nan: only at start & end
                (task.Prog.acc_maskwise, 50),
                (monitor.Prog.loss_log, 50),
                (monitor.Prog.acc_log, 50),
                (monitor.Prog.LA_plot, 500),
                (task.Param.P_max_mean, 50),
                #monitor.Param.P_max_mean_log, 100),
                (monitor.Param.P_max_mean_window, 500),
                (monitor.X.out_2d, nan),
                (monitor.Param.P_show, nan),
              ],
    streaming = True,
    #save_dir = f"{cwd}/results_node_train_0_cuda",
)
result.learn(total_epoch=5000)

sig_A = t.sigmoid(100*t.matmul(t.transpose(node_train_model.A,0,1),node_train_model.A))
print(f"Best fitting Adj: {node_train_model.A}")
utils.P_draw(sig_A,size=(15,7)); plt.show()
print(f"Binary A maximum: {t.sigmoid(100*node_train_model.A).max()}")
print(f"Binary A minimum: {t.sigmoid(100*node_train_model.A).min()}")


#--------------------------------------------------------------------------------------------#
# binarizing A and checking accuracy

bin_A = t.zeros(D.num_train,D.num_train,dtype=t.int,device=0)
for i in range(D.num_train):
    for j in range(D.num_train):
        if sig_A[i,j] > 0.5:
            bin_A[i,j] = 1
        else:
            bin_A[i,j] = 0

plt.matshow(bin_A.detach().cpu(),cmap='jet'); plt.show()

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
        x = t.tensordot(self.A/D.num_train,x,dims=[[0],[0]])
        x = t.tensordot(x,self.W0,dims=[[1],[0]])+self.b0
        x = F.sigmoid(x)
        x = t.tensordot(self.A/D.num_train,x,dims=[[1],[0]])
        x = t.tensordot(x,self.W1,dims=[[1],[0]])+self.b1
        return x

pipeline.BasicTest(model = NodeClassifier_test(A=bin_A),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = D.x[D.train_mask], y = D.y[D.train_mask] )


#-------------------------------------------------------------------------------------------#
# binary extrapolated A test

ind = D.num_classes*[None]
for i in range(0,D.num_classes):
    ind[i] = D.y[D.train_mask] == i

trained_A_cluster = np.zeros((D.num_classes, D.num_classes))

for i in range(0,D.num_classes):
    for j in range(0,D.num_classes):
        trained_A_cluster[i,j] =  t.round(sig_A[ind[i]][:,ind[j]].mean())
    
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

extrapolated_A_noise = t.tensor(extrapolated_A+0.2*np.random.randn(D.num_test,D.num_test)).float()
utils.P_draw(extrapolated_A_noise); plt.show()


print("Extrapolated A test set result:")
pipeline.BasicTest(model = NodeClassifier_test(A=extrapolated_A.cuda()/D.num_train),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = t.transpose(D.x[D.test_mask],0,1), y = D.y[D.test_mask])

print("Extrapolated A_noise test set result:")
pipeline.BasicTest(model = NodeClassifier_test(A=extrapolated_A_noise.cuda()/D.num_train),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = t.transpose(D.x[D.test_mask],0,1), y = D.y[D.test_mask])



#-------------------------------------------------------------------------------------------#
# processing for binarized edge learning

edge_y = t.zeros(D.num_train,D.num_train, dtype=t.long, device=0)
edge_x = t.zeros(D.num_train,D.num_train,2*D.num_features, device=0)
#D.unload_cuda()

for i in range(0,D.num_train):
    for j in range(0,D.num_train):
        edge_y[i,j] = sig_A[i,j].item()
        edge_x[i,j,:] = t.cat( (D.x[D.train_mask][i,:],D.x[D.train_mask][j,:]), dim=0 )
    
edge_x = edge_x.view(-1,2*D.num_features)
edge_y = edge_y.view(-1,)

edge_train_mask, edge_val_mask, edge_test_mask = \
  datatool.masker(edge_x.shape[0],p_train=0.7,p_val=0.3,p_test=0)
  
edge_D = datatool.sortedDF(edge_x,edge_y, 
                           edge_train_mask, edge_val_mask, edge_test_mask, sort_class=False)

edge_D.load_cuda()

  
  
#-------------------------------------------------------------------------------------------#
# performing edge learning

class EdgePredictor(t.nn.Module):
    def __init__(self,seed,reg_x,hidden_channels):
        super().__init__()
        t.manual_seed(seed); 

        self.W0 = P(t.randn(reg_x.shape[1],hidden_channels[0]))
        self.b0 = P(t.randn(hidden_channels[0]))
        self.W1 = P(t.randn(hidden_channels[0],hidden_channels[1]))
        self.b1 = P(t.randn(hidden_channels[1]))
        self.W2 = P(t.randn(hidden_channels[1],2))
        self.b2 = P(t.randn(2))
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
edge_train_model = EdgePredictor(seed, edge_x, hidden_channels=[200,100] ).cuda()
criterion = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(edge_train_model.parameters(), lr=5e-3, weight_decay=1e-5)
total_epoch = 100000
train_x = edge_D.x[edge_D.train_mask]; train_y = edge_D.y[edge_D.train_mask]
val_x=edge_D.x[edge_D.val_mask]; val_y=edge_D.y[edge_D.val_mask]; test_x=[]; test_y=[]
tasklist = [ 
            (task.Prog.loss_basic, 1), # int: per | nan: only at end
            (task.Prog.acc_maskwise, 100),
            (monitor.Prog.loss_log, 50),
            (monitor.Prog.acc_log, 100),
            (monitor.Prog.LA_plot, 2000),
            (task.Param.P_max_mean, 100),
            #(monitor.Param.P_max_mean_log, 1e10),
            (monitor.Param.P_max_mean_window, 2000),
            #(monitor.X.out_2d, nan),
            (monitor.Param.P_show, nan),
          ]
streaming = True
save_dir = "c:/Users/echo_/Research/GNN_graph_infer/results_edge_train_binary_1",
result = pipeline.BasicLearn(edge_train_model,D,criterion,optimizer,total_epoch,train_x,train_y,
                             tasklist,val_x,val_y,test_x,test_y,streaming)


#--------------------------------------------------------------------------------------------------#
# drawing reconstructed edge weights

edge_train_model.eval()

edge_out_tr_val = t.squeeze(edge_train_model.forward(edge_x))
edge_out_tr_val = edge_out_tr_val.argmax(dim=1).view(D.num_train,D.num_train)
print("Best fitting weights & Reconstructed weights:")
utils.P_draw([sig_A, edge_out_tr_val]); plt.show()

edge_x_test = t.zeros(D.num_test,D.num_test,2*D.num_features)
for i in range(D.num_test):
    for j in range(D.num_test):
        edge_x_test[i,j,:] = t.cat( (D.x[D.test_mask][i,:],D.x[D.test_mask][j,:]), dim=0 )
edge_x_test = edge_x_test.view(-1,2*D.num_features).cuda()
    
edge_out_test = edge_train_model.forward(edge_x_test)
edge_out_test = edge_out_test.argmax(dim=1).view(D.num_test,D.num_test)
print("Extrapolated test set weights:")
utils.P_draw(edge_out_test); plt.show()

#edge_x_full = t.zeros(D.num_nodes,D.num_nodes,2*D.num_features)


#--------------------------------------------------------------------------------------------------#
# checking final results

class NodeClassifier_test(t.nn.Module):
    def __init__(self, A):
        super().__init__()
        t.manual_seed(seed)
        
        self.A = A
        self.W0 = node_train_model.W0
        self.b0 = node_train_model.b0
        self.W1 = node_train_model.W1
        self.b1 = node_train_model.b1
        self.params = [self.A,self.W0,self.b0,self.W1,self.b1]

    def forward(self, x):
        x = t.tensordot(x,self.A,dims=[[1],[0]])
        x = t.tensordot(self.W0,x,dims=[[1],[0]])+self.b0
        x = t.sigmoid(x)
        x = t.tensordot(x,self.A,dims=[[1],[0]])
        x = t.tensordot(self.W1,x,dims=[[1],[0]])+self.b1
        return t.transpose(x,0,1)
  
pipeline.BasicTest(model = NodeClassifier_test(A=edge_out_tr_val/D.num_train),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = t.transpose(D.x[D.train_mask],0,1), y = D.y[D.train_mask])
  
pipeline.BasicTest(model = NodeClassifier_test(A=edge_out_test/D.num_test),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = t.transpose(D.x[D.test_mask],0,1), y = D.y[D.test_mask])



#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
# processing for binarized edge learning 2

edge_y = t.zeros(D.num_train,D.num_train, dtype=t.long, device=0)
edge_x = t.zeros(D.num_train,D.num_train,2*D.num_features, device=0)

for i in range(0,D.num_train):
    for j in range(0,D.num_train):
        edge_y[i,j] = bin_A[i,j].item()
        edge_x[i,j,:] = t.cat( (D.x[D.train_mask][i,:],D.x[D.train_mask][j,:]), dim=0 )
    
edge_x = edge_x.view(-1,2*D.num_features)
edge_y = edge_y.view(-1,)

etm, evm, _, = datatool.masker(D.num_train,p_train=0.7,p_val=0.3,p_test=0)
etm.cuda(); evm.cuda()

edge_y_tr = t.zeros(etm.sum(),etm.sum(), dtype=t.long, device=0)
edge_x_tr = t.zeros(etm.sum(),etm.sum(), 2*D.num_features, device=0)
edge_y_val = t.zeros(evm.sum(),evm.sum(), dtype=t.long, device=0)
edge_x_val = t.zeros(evm.sum(),evm.sum(), 2*D.num_features, device=0)

for i in range(0,etm.sum()):
    for j in range(0,etm.sum()):
        edge_y_tr[i,j] = bin_A[etm][:,etm][i,j].item()
        edge_x_tr[i,j,:] = t.cat( (D.x[D.train_mask][etm][i,:],D.x[D.train_mask][etm][j,:]), dim=0 )
    
for i in range(0,evm.sum()):
    for j in range(0,evm.sum()):
        edge_y_val[i,j] = bin_A[evm][:,evm][i,j].item()
        edge_x_val[i,j,:] = t.cat( (D.x[D.train_mask][evm][i,:],D.x[D.train_mask][evm][j,:]), dim=0 )
    
edge_x_tr = edge_x_tr.view(-1,2*D.num_features)
edge_y_tr = edge_y_tr.view(-1,)
edge_x_val = edge_x_val.view(-1,2*D.num_features)
edge_y_val = edge_y_val.view(-1,)


#-------------------------------------------------------------------------------------------#
# performing edge learning

class EdgePredictor(t.nn.Module):
    def __init__(self,seed,reg_x,hidden_channels):
        super().__init__()
        t.manual_seed(seed); 

        self.W0 = P(t.randn(reg_x.shape[1],hidden_channels[0]))
        self.b0 = P(t.randn(hidden_channels[0]))
        self.W1 = P(t.randn(hidden_channels[0],hidden_channels[1]))
        self.b1 = P(t.randn(hidden_channels[1]))
        self.W2 = P(t.randn(hidden_channels[1],2))
        self.b2 = P(t.randn(2))
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
edge_train_model = EdgePredictor(seed, edge_x_tr, hidden_channels=[200,100] ).cuda()
criterion = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(edge_train_model.parameters(), lr=5e-3, weight_decay=1e-5)
train_x = edge_x_tr; train_y = edge_y_tr
val_x=edge_x_val; val_y=edge_y_val; test_x=[]; test_y=[]
tasklist = [ 
            (task.Prog.loss_basic, 1), # int: per | nan: only at end
            (task.Prog.acc_maskwise, 10),
            (monitor.Prog.loss_log, 20),
            (monitor.Prog.acc_log, 20),
            (monitor.Prog.LA_plot, 2000),
            (task.Param.P_max_mean, 100),
            #(monitor.Param.P_max_mean_log, 1e10),
            (monitor.Param.P_max_mean_window, 2000),
            #(monitor.X.out_2d, nan),
            (monitor.Param.P_show, nan),
          ]
streaming = True
save_dir = "c:/Users/echo_/Research/GNN_graph_infer/results_edge_train_binary_1",
result = pipeline.BasicLearn(edge_train_model,D,criterion,optimizer,
                             train_x,train_y,val_x,val_y,test_x,test_y,tasklist,streaming)
result.learn(total_epoch=2000)


#--------------------------------------------------------------------------------------------------#
# drawing reconstructed edge weights

edge_train_model.eval()

edge_out_tr_val = t.squeeze(edge_train_model(edge_x))
edge_out_tr_val = edge_out_tr_val.argmax(dim=1).view(D.num_train,D.num_train)
print("Best fitting weights & Reconstructed weights:")
utils.P_draw([bin_A, edge_out_tr_val]); plt.show()

edge_x_test = t.zeros(D.num_test,D.num_test,2*D.num_features)
for i in range(D.num_test):
    for j in range(D.num_test):
        edge_x_test[i,j,:] = t.cat( (D.x[D.test_mask][i,:],D.x[D.test_mask][j,:]), dim=0 )
edge_x_test = edge_x_test.view(-1,2*D.num_features).cuda()
    
edge_out_test = edge_train_model.forward(edge_x_test)
edge_out_test = edge_out_test.argmax(dim=1).view(D.num_test,D.num_test)
print("Extrapolated test set weights:")
utils.P_draw(edge_out_test); plt.show()

#edge_x_full = t.zeros(D.num_nodes,D.num_nodes,2*D.num_features)


#--------------------------------------------------------------------------------------------------#
# checking final results

class NodeClassifier_test(t.nn.Module):
    def __init__(self, A):
        super().__init__()
        t.manual_seed(seed)
        
        self.A = A
        self.W0 = node_train_model.W0
        self.b0 = node_train_model.b0
        self.W1 = node_train_model.W1
        self.b1 = node_train_model.b1
        self.params = [self.A,self.W0,self.b0,self.W1,self.b1]

    def forward(self, x):
        x = t.tensordot(self.A,x,dims=[[0],[0]])
        x = t.tensordot(x,self.W0,dims=[[1],[0]])+self.b0
        x = t.sigmoid(x)
        x = t.tensordot(self.A,x,dims=[[0],[0]])
        x = t.tensordot(x,self.W1,dims=[[1],[0]])+self.b1
        return x

pipeline.BasicTest(model = NodeClassifier_test(A=edge_out_tr_val/D.num_train),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = D.x[D.train_mask], y = D.y[D.train_mask])
  
pipeline.BasicTest(model = NodeClassifier_test(A=edge_out_test/D.num_test),
                   criterion = t.nn.CrossEntropyLoss(),
                   x = D.x[D.test_mask], y = D.y[D.test_mask])
