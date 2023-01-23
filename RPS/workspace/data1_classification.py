import os; cwd = os.getcwd()+"/"
from itertools import compress
from torch_utils import pipeline, nnframe, datatool
import torch as t
from torch.nn import Linear, Identity
from torch_geometric.nn import TopKPooling, GCNConv
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np; import numba as nb
from joblib import Parallel, delayed


# preprocessing data -------------------------------------------------------------------------
categories = ['L50_0.1N','L50_1N','L50_2N','L50_5N','L100_0.1N','L100_1N']
categories_ = ['L50_s0.1N_f10N','L50_s1N_f10N','L50_s2N_f10N','L50_s5N_f10N',
               'L100_s0.1N_f4N','L100_s1N_f4N']
sizes = 4*[50]+2*[100]
category = 2; samples = 2

X_ = np.zeros((samples,sizes[category]**2))

def load_category_sample(i):
    X_[i] = np.array(pd.read_csv(f"lattice_data/{categories[category]}_{i+1}.csv",header=None))\
            .reshape(-1)
            
def load_category(category,samples):
    Parallel(n_jobs=24,require='sharedmem')( delayed(load_category_sample)(i) for i in range(samples) )
    Y = np.array(pd.read_csv(f"lattice_data/{categories_[category]}_summary.csv",header=None)\
          [:samples]).squeeze()
    return Y

Y = load_category(category,samples); Y = t.tensor(Y)
X = np.zeros((samples,sizes[category]**2,3)); G = []

def build_feature_sample(i,sample):
    #! building feature
    for j, specimen in enumerate(sample):
        if specimen == 1:
            X[i,j] = np.array([1.,0.,0.])
        elif specimen == 2:
            X[i,j] = np.array([0.,1.,0.])
        elif specimen == 3:
            X[i,j] = np.array([0.,0.,1.])
    #! building Graph
        #G_ = t.eye(10,10).to_sparse().indices()
        #G.append(G_)

def build_feature(X):
    Parallel(n_jobs=24,require='sharedmem')\
        ( delayed(build_feature_sample)(i,X_[i]) for i in range(samples) )

build_feature(X); X = t.tensor(X); 

trm, tm = datatool.splitter(samples,p=0.7)
data = datatool.DataStruct(X[trm],Y[trm],list(compress(G,trm.tolist())))
data.set_test(X[tm],Y[tm],list(compress(G,trm.tolist())))
setattr(data,'lattice_size',sizes[category])


# training ------------------------------------------------------------------------------------
pool = 1
model = nnframe.GNN(layers = [TopKPooling, Linear],
                    args = [{'in_channels':data.lattice_size**2,'ratio':pool},
                            {'in_features':data.num_features,'out_features':data.num_targets}],
                    # layers = [GCNConv, Linear],
                    # args = [[3,3],[3,2]],
                    activations = [t.sigmoid, Identity()],)

model(data.x_train[0],data.G_train[0])

def train_batch(self):
    for self.x_batch, self.y_batch, self.G_batch in self.trainloader: # grabbing batch
        self.x_batch, self.y_batch = \
            self.x_batch.to(self.device), self.y_batch.to(self.device)
        print(self.x_batch,self.y_batch,self.G_batch)
        self.model.train()
        self.train_out_batch = self.model(self.x_batch,self.G_batch)
        self.train_loss_batch = self.criterion(self.train_out_batch,self.y_batch)
      
        if self.status is True: # computing gradients for each mini-batch
            self.optimizer.zero_grad()
            self.train_loss_batch.backward(retain_graph = self.retain_graph) #! BACKWARD ROUTINE
            self.optimizer.step()

class RPS_GNN(pipeline.Simple):
    def __init__(self,*args):
        super().__init__(*args)
    
    def map_task(self):
        self.train_batch = train_batch
    
    def set_data(self):
        if self.batch_size is not False:
            self.batch_size = self.batch_size
        else:
            self.batch_size = self.data.num_train
        self.trainloader =\
            DataLoader(self.data,batch_size=self.batch_size,shuffle=self.shuffle,pin_memory=self.gpu)
        self.x_train = self.data.x_train; self.y_train = self.data.y_train
        self.x_val = self.data.x_val; self.y_val = self.data.y_val
        self.x_test = self.data.x_test; self.y_test = self.data.y_test

work = RPS_GNN(data,model)
work.learn( criterion = t.nn.CrossEntropyLoss(),
            optimizer = t.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5),
            total_epoch=100, batch_size=1, streaming=True, gpu=False,)