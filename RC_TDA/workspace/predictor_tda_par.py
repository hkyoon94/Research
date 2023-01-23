import os, dill; from time import time;
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from pandas import DataFrame as DF
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio; pio.renderers.default = "vscode"
from gtda.homology import VietorisRipsPersistence as VRP
from gtda.diagrams import PairwiseDistance
from gtda.plotting import plot_point_cloud, plot_diagram
from joblib import Parallel, delayed  
from routines import cwd, mkdir, saveobj, loadobj, pointwise_distance, ESN


def main(i,n,traj):
  
    print(f"\tNow in data {i}, window {n}...")
    mkdir(f'{masterdir}/window_{n}/outs')
    mkdir(f'{masterdir}/window_{n}/persistence')
    
    
    # reservoir computing
    for i in range(1,predictor_num+1):
        L = len(traj)
        PreLen = 0; InitLen = int(0.2*L); TrainLen = int(0.4*L); TestLen = int(0.4*L)
        
        traj_ = traj + 0.001*np.random.randn(*traj.shape)
        
        Res = ESN(N,p,rho,leak,bias,factor)
        Res.record(traj_,PreLen,InitLen,TrainLen)
        Res.optimize('pinv_reg',1e-4)
        Res.get_train_error(disp=False)
        Res.generate_out(TestLen,record_state=False,record_feedback_ratio=False,record_proj_ratio=False)
        Res.get_test_error(disp=False)
        fig1, fig2 = Res.render_out()
        #fig3 = Res.plot_singular_values()
        #fig1.write_html(f'{masterdir}/window_{n}/outs/res{i}_phasespace.html')
        fig1.write_image(f'{masterdir}/window_{n}/outs/res{i}_phasespace.png')
        fig2.write_image(f'{masterdir}/window_{n}/outs/res{i}_xyz.png')
        
        saveobj(Res.out.T,f'{masterdir}/window_{n}/outs/res{i}.pkl')  
    saveobj(Res.target,f'{masterdir}/window_{n}/target.pkl')


    # loading reservoir out data
    target_ = loadobj(f'{masterdir}/window_{n}/target.pkl').T[::TDA_jump]
    X = np.zeros((predictor_num+1,target_.shape[0],target_.shape[1]))
    X[0,:,:] = target_
    for i in range(1,predictor_num+1):
        X[i,:,:] = loadobj(f'{masterdir}/window_{n}/outs/res{i}.pkl')[::TDA_jump]
        #plot_point_cloud(X[i,:,:]).write_image(f'{masterdir}/window_{n}/persistence/res{i}_ptcloud.png')
    
  
    # performing TDA & computing pairwise homology distance
    persistence = VRP(metric="euclidean", homology_dimensions=(0,1,2), collapse_edges=True,n_jobs=1)
    X_homology = persistence.fit_transform(X) # TDA

    for i in range(1,predictor_num+1):
        plot_diagram(X_homology[i]).write_image(f'{masterdir}/window_{n}/persistence/res{i}.png')

    X_distance = PairwiseDistance(metric='landscape',n_jobs=1).fit_transform(X_homology)
    plt.matshow(X_distance); plt.colorbar()
    plt.savefig(f'{masterdir}/window_{n}/pairwise_TDA_distance.png'); plt.close()
    
    predictor_mean = X_distance[1:,1:].sum().sum()/(predictor_num**2)
    
    with open(f'{masterdir}/homology_dist_mean.txt','a') as f:
        np.savetxt(f,[np.array([n,predictor_mean])],delimiter=',')
    
    
    # computing pointwise-variance
    X_distance_ptwise = np.zeros((predictor_num+1,predictor_num+1))
    for i in range(predictor_num+1):
        for j in range(predictor_num+1):
            X_distance_ptwise[i,j] = pointwise_distance(X[i,:,:],X[j,:,:])
        
    plt.matshow(X_distance_ptwise); plt.colorbar()
    plt.savefig(f'{masterdir}/window_{n}/pairwise_point_distance.png'); plt.close()
    
    predictor_mean_ptwise = X_distance_ptwise[1:,1:].sum().sum()/(predictor_num**2)
    
    with open(f'{masterdir}/pointwise_dist_mean.txt','a') as f:
        np.savetxt(f,[np.array([n,predictor_mean_ptwise])],delimiter=',')


#----------------------- master task -------------------------#

##### GLOBAL PARAMETERS
window_interval = 3000; window_num = 200; data_jump = 3; predictor_num = 50; TDA_jump = 7
N = 300; p = 0.25; rho=0.08; leak=0.05; bias=0.05; factor=1

for i in range(4):

    print(f'Now in data {i}')
    masterdir = f'Rossler_coupled_results{i}'

    #_, data = loadobj('data/Rossler_traj.pkl')
    data = np.loadtxt(f'{cwd}/data/data{i}.txt',delimiter=',')
    #data = loadobj('data/Rossler_bifur_traj.pkl')
    full_traj = data[:,0:3]
    r = data[:,-1]

    window_start = np.linspace(0,full_traj.shape[0]-window_interval,window_num,dtype=int)
    r_ = np.take(r,window_start)

    def get_traj(n):
      return full_traj[window_start[n]:window_start[n]+window_interval:data_jump]

    ts = time()
    Parallel(n_jobs=24)(delayed(main)(i,n,get_traj(n)) for n in range(window_num))
    tf = time()
    print(f'Data {i}: took {tf-ts:.4f} seconds')

    pm = np.loadtxt(f'{masterdir}/homology_dist_mean.txt',delimiter=',')
    pm = pm[pm[:,0].argsort()]
    px.line(x=r_,y=pm[:,1]).write_image(f'{masterdir}/mean_homology_dist_data{i}.png')

    pmp = np.loadtxt(f'{masterdir}/pointwise_dist_mean.txt',delimiter=',')
    pmp = pmp[pmp[:,0].argsort()]
    px.line(x=r_,y=pmp[:,1]).write_image(f'{masterdir}/mean_pointwise_dist_data{i}.png')