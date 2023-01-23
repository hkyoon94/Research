import time, json
from collections import OrderedDict
import torch
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from numpy.random import uniform, randint
from joblib import Parallel, delayed
from math import nan
from RPS_routines import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


L = 100; N = L**2
T = int(1*N)
samples = 20
score_interval = 1

mobilities = uniform(1e-4,1e-3,samples)
prob_params = produce_prob_params(mobilities,rho=1,k=1,L=L)

scores = np.zeros((samples,int(T/score_interval),3))
final_stats = np.zeros(samples)

for sample_id in range(0,samples):
    
    final_status = 0
    sample_scores = np.zeros(scores.shape[1:])
    score_ct = -1
    
    Lattice = randint(1,4, (L,L))
    st = time.time()
    for t in range(0,T):
        Lattice = simulate_one_generation(Lattice, L, *prob_params[sample_id])
        if t % score_interval == 0:
            score_ct += 1
            sample_scores[score_ct,:] = measure_score(Lattice)
            #sample_scores[score_ct,:,:,:] = measure_block_score(Lattice,block_size)
    
    print(f"Elapsed time: {time.time()-st}")
    scores[sample_id,:,:] = sample_scores
    
    density = measure_density(Lattice,N)
    
    if not (density<0.0001).sum() == 0:
        final_stats[sample_id] = 1


fig = go.Figure()
plot_length = 10000
for sample_id in range(0,samples):
    if final_stats[sample_id] == 0:
        fig.add_traces(data=go.Scatter3d(
            x=scores[sample_id,0:plot_length,0], 
            y=scores[sample_id,0:plot_length,1], 
            z=scores[sample_id,0:plot_length,2],
            mode='lines', line_color='green'))
    else:
        fig.add_traces(data=go.Scatter3d(
            x=scores[sample_id,0:plot_length,0], 
            y=scores[sample_id,0:plot_length,1], 
            z=scores[sample_id,0:plot_length,2],
            mode='lines', line_color='red'))
fig.write_html('fig.html')


deriv_norms = np.zeros((samples,len(scores[0])-1))
for sample_id in range(0,samples):
    deriv_norms[sample_id] = score_time_derivative_norm(scores[sample_id])
    
fig = go.Figure()
for sample_id in range(0,samples):
    if final_stats[sample_id] == 0:
        fig.add_traces(data=go.Line(
            y=deriv_norms[sample_id],line_color='green'))
    else:
        fig.add_traces(data=go.Line(
            y=deriv_norms[sample_id],line_color='red'))
fig.write_html('fig2.html')


#----------------------------------------------------------------------------------------

def load_lattice(id):
    return pd.read_csv(f"stats200/L200_0.1N_{id}.csv",header=None).values

samples = 5000
X = np.array( Parallel(n_jobs=24)\
        ( delayed(load_lattice)(id) for id in range(1,samples+1) ) )

scores = np.array( Parallel(n_jobs=24)\
        ( delayed(measure_score)(X[id]) for id in range(samples) ) )

mobilities_Y = pd.read_csv('stats200/L200_f10N_summary.csv',header=None).values

mobilities = mobilities_Y[:,0]
Y = mobilities_Y[:,1]

Z = np.array(samples*[None])
Z_ = np.array(samples*[None])
for i in range(samples):
    m = mobilities[i]
    if m >= 1e-3 and m < 1e-2:
        Z[i] = '1e-3=<ep<1e-2'
        Z_[i] = 0
    elif m >= 1e-4 and m < 1e-3:
        Z[i] = '1e-4=<ep<1e-3'
        Z_[i] = 1
    elif m >= 1e-5 and m < 1e-4:
        Z[i] = '1e-5=<ep<1e-4'
        Z_[i] = 2
    elif m >= 1e-6 and m < 1e-7:
        Z[i] = '1e-6=<ep<1e-7'
        Z_[i] = 3
    else:
        Z[i] = 'ep<1e-6'
        Z_[i] = 4
        
filt = np.array(samples*[False])
for i in range(samples):
    m = mobilities[i]
    if m >= 5*1e-4 and m < 8*1e-4:
        filt[i] = True  

scores = scores[filt]
Z = Z[filt]
Y = Y[filt]

px.scatter_3d(pd.DataFrame(scores,columns=('x','y','z')),x='x',y='y',z='z',color=Z)\
    .update_coloraxes(showscale=False)\
    .update_traces(marker_size=2)\
    .write_html(f'score_mobility.html')

px.scatter_3d(pd.DataFrame(scores,columns=('x','y','z')),x='x',y='y',z='z',color=Y,\
    color_continuous_scale=['green','red'])\
    .update_coloraxes(showscale=False)\
    .update_traces(marker_size=4)\
    .write_html(f'score_Y.html')
    
    
scores_TSNE = TSNE(n_components=2, learning_rate='auto', 
                  init='random', perplexity=3).fit_transform(scores)
fig, ax = plt.subplots()
ax.scatter(scores_TSNE[:,0], scores_TSNE[:,1],c=Z_,s=3,cmap='jet',label=Z)
plt.legend()
plt.show(fig)