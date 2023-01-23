import os, time; cwd = os.getcwd()+"/"
from torch_utils import pipeline, nnframe, datatool
import torch
import pandas as pd
import plotly.express as px
import numpy as np
from joblib import Parallel, delayed
from math import nan
from RPS_routines import measure_score

categories = ['L50_0.1N','L50_1N','L50_2N','L50_5N','L100_0.1N','L100_1N']
categories_ = ['L50_s0.1N_f10N','L50_s1N_f10N','L50_s2N_f10N','L50_s5N_f10N',
               'L100_s0.1N_f4N','L100_s1N_f4N']
sizes = 4*[50]+2*[100]

def load_lattice(id):
    return np.array(pd.read_csv(f"lattice_data/{categories[category]}_{id+1}.csv",header=None))


def main(category,samples):
  
    print(f"\n\n-------- Now in category: {categories[category]} --------\n")
    print(f"Loading {samples} samples..."); ts = time.time()
    
    X = np.array( Parallel(n_jobs=24)\
        ( delayed(load_lattice)(id) for id in range(samples) ) )
    
    Y = np.array( pd.read_csv(f"lattice_data/{categories_[category]}_summary.csv",header=None)\
                    [:samples] ).squeeze()
    
    print(f"\ttook {time.time()-ts:6f} seconds.\n")
    
    print(f"Extracting species features..."); ts = time.time()
    
    U = np.array( Parallel(n_jobs=24)\
        ( delayed(measure_score)(X[id]) for id in range(samples) ) )
    
    print(f"\ttook {time.time()-ts:6f} seconds.\n")
    
    px.scatter_3d(pd.DataFrame(U,columns=('x','y','z')),x='x',y='y',z='z',color=Y,
                    color_continuous_scale=['green','red'])\
                    .update_coloraxes(showscale=False).update_traces(marker_size=2)\
                        .write_html(f'score_{categories[category]}.html')

    U = torch.tensor(U,dtype=torch.float)
    Y = torch.tensor(Y,dtype=torch.long)
    
    trm, tm = datatool.splitter(samples,p=0.9,name1='train set',name2='test set',seed=3)
    data = datatool.DataStruct(U[trm],Y[trm])
    data.set_test(U[tm],Y[tm])

    model = nnframe.FFNN(
        layers = 2*[torch.nn.Linear],
        args = [
            {'in_features':data.num_features, 'out_features':16},
            {'in_features':16, 'out_features':data.num_targets}
        ],
        activations = 2*[torch.relu],
        dropouts = [0.5,0.]
    )

    work = pipeline.Builder(data,model)
    work.learn(
        criterion = torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5),
        tasklist = [
            (work.loss,1), 
            (work.acc_maskwise,1),
            (work.loss_log,1), 
            (work.acc_log,1), 
            (work.LA_plot,nan),
            ],
            total_epoch=40, batch_size=500, 
            forward_test=True, streaming=True, gpu=False,
    )

for category in range(len(categories)):
    main(category, samples=20000)


#------------------------------------------------------------------------------------------
    
samples = 20000
category = 4
X = np.array( Parallel(n_jobs=24)\
        ( delayed(load_lattice)(id) for id in range(samples) ) )

U = np.array( Parallel(n_jobs=24)\
        ( delayed(measure_score)(X[id]) for id in range(samples) ) )

Y = np.array( pd.read_csv(f"lattice_data/{categories_[category]}_summary.csv",header=None)\
                    [:samples] ).squeeze()

px.scatter_3d(pd.DataFrame(U,columns=('x','y','z')),x='x',y='y',z='z',color=Y,
                    color_continuous_scale=['green','red'])\
                    .update_coloraxes(showscale=False).update_traces(marker_size=2)\
                        .show()
                        #.write_html(f'score_{categories[category]}.html')

