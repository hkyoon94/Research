import os, dill; from time import time
import numpy as np
import plotly.express as px
from pandas import DataFrame as DF
import plotly.io as pio; pio.renderers.default = "vscode"
import matplotlib.pyplot as plt
from routines import mkdir, saveobj, loadobj, ESN

#------------------------------------ for degrad test ------------------------------------#

# #_, traj = loadobj('data/Rossler_traj.pkl')
# traj = loadobj('data/Rossler_bifur_traj.pkl')
# traj = traj[300000:400000:10,0:3]
# traj_ = traj + 0.001*np.random.randn(*traj.shape)
# L = len(traj_)
# PreLen = 0; InitLen = int(0.2*L); TrainLen = int(0.7*L); TestLen = int(0.1*L)

# ratio = np.linspace(0,1,4)[1:-1]

# N = 500; p = 20/N; rho=0.17; leak=1; bias=0.01; factor=1
# rule = ESN.rule1

# Res = ESN(N,p,rho,leak,bias,factor)
# Res.record(traj_,rule,PreLen,InitLen,TrainLen)
# Res.optimize_pinv_degrad(ratio)
# Res.disp_train_error()
# Res.generate_out(TestLen)
# fig1, fig2 = Res.render_out()
# fig1.show(); fig2.show()

#------------------------------------ for eigval test ------------------------------------#

#_, traj = loadobj('data/Rossler_traj.pkl')
traj = loadobj('data/Rossler_bifur_traj.pkl')
traj = traj[200000:350000:10,0:2]; #traj_ = traj
traj_ = traj + 0.0001*np.random.rand(*traj.shape)
L = len(traj_)
PreLen = 0; InitLen = int(0.2*L); TrainLen = int(0.3*L); TestLen = int(0.2*L)

N = 500; p = 20/N; rho=0.17; leak=1; bias=0.01; factor=1
rule = ESN.rule1

Res = ESN(N,p,rho,leak,bias,factor)
Res.record(traj_,rule,PreLen,InitLen,TrainLen)
Res.optimize_pinv()
Res.get_train_error(disp=True)
Res.generate_out(TestLen,record_state=True)
Res.get_test_error(disp=True)
fig1, fig2 = Res.render_out()
fig1.show(); fig2.show()
Res.check_blackhole(n_eig_space=3)
px.line(DF(Res.danger_)).show()

#A_eigs = np.linalg.eigvals(Res.A)
#W_eigs = np.linalg.eigvals(Res.Wout@Res.Win)
AplusW_eigs = np.linalg.eigvals(Res.A+Res.Wout@Res.Win)
#_, Win_sv, _ = np.linalg.svd(Res.Win)
#_, Wout_sv, _ = np.linalg.svd(Res.Wout)

fig, ax = plt.subplots(2,3)
x = np.arange(0,10)
ax[0,0].bar(x,np.real(AplusW_eigs[:10]))
fig.show()
#ax[0,1].bar(x,np.sort(Res.Wout.view()))
