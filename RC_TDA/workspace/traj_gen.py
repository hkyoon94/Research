import os, dill; from time import time
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame as DF
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio; pio.renderers.default = "vscode"
from routines import RK4, RK4_bifur, rossler, mkdir, saveobj, loadobj

iniy = np.array([1.,1.,1.]); tfin=5000; h=0.01
t = np.arange(0,tfin+0.1*h,h)
c_ = np.linspace(1.,8.,t.shape[0]); a = 0.2; b = 0.4
_, soln = RK4_bifur(rossler, iniy,tfin,h, c_,a,b)
solnc_ = np.column_stack((soln,c_))
#px.line_3d(DF(soln[::10]),x=0,y=1,z=2).show()

mkdir('data')
saveobj(solnc_,'data/Rossler_bifur_traj.pkl')

traj = loadobj('data/Rossler_bifur_traj.pkl')
for n in range(int(traj.shape[0]/10000)):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(f'c={traj[10000*n]} to c={traj[10000*(n+1)]}')
    ax.plot3D(traj[10000*n:10000*(n+1):20,0],traj[10000*n:10000*(n+1):20,1],\
        traj[10000*n:10000*(n+1):20,2]); fig.show()