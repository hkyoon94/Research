import os, dill; from time import time
from varname import nameof
import numpy as np
from numpy import *
from numpy.random import *
from numpy.linalg import *
import plotly.express as px
from pandas import DataFrame as DF
import plotly.io as pio; pio.renderers.default = "vscode"
import matplotlib.pyplot as plt
from routines import cwd, mkdir, saveobj, loadobj, id, ESN


traj = loadobj('data/Rossler_bifur_traj.pkl')
#traj = np.loadtxt(f'{cwd}/data/data{1}.txt',delimiter=',')
traj = traj[400000:450000:5,0:3]
#traj_ = traj
traj_ = traj + 0.0001*rand(*traj.shape)
L = len(traj_)
PreLen = 0; InitLen = int(0.2*L); TrainLen = int(0.4*L); TestLen = int(0.4*L)

N = 300; p = 0.25; rho=0.08; leak=0.05; bias=0.05; factor=1

Res = ESN(N,p,rho,leak,bias,factor)
Res.record(traj_,PreLen,InitLen,TrainLen)
Res.optimize('pinv_reg',1e-4)
Res.get_train_error(disp=True)
#Res.stabilize(2,0.001)
Res.generate_out(TestLen,record_state=True,record_feedback_ratio=True,record_proj_ratio=True)
Res.get_test_error(disp=True)
fig1, fig2 = Res.render_out()
fig1.show(); fig2.show()
#Res.check_blackhole(n_eig_space=3)

fig,ax = plt.subplots()
x = arange(0,10)
ax.bar(x,abs(Res.S[:10]))
ax.set_title('10 Largest values of |sig(A)|')
plt.show(fig)

# px.line(DF(Res.feedback_ratio_)).show()
# px.line(DF(Res.proj_ratio_)).show()

# D_abs = abs(D)
# D_abs_ind = argsort(-abs(D))
# D = D[D_abs_ind]; Vh = Vh[:,D_abs_ind]
# i=0
# while True:
#     if D_abs[i] == D_abs[i+1]:
#         break
#     else:
#         i+=1
# vr = real(V[:,i]); vi = imag(V[:,i])
# lr = real(D[i]); li = imag(D[i])

# iternum = 1000; N = Res.N
# transform_ratio = zeros((iternum,2))
# for k in range(iternum):
#     x = randn(N)
#     transform_ratio[k,0] = projnorm_S(x,V[:,0],V[:,1],V[:,2])/norm(x)
#     transform_ratio[k,1] = norm(A@x)/norm(x)

# px.scatter(DF(transform_ratio,columns=['projection ratio','transform ratio']), 
#         x='projection ratio',y='transform ratio')



#---------------------------------------------------------------------------------------------------#
# projnorm(x,vr,vi)/norm(x,ord=2)
# y = A@x
# norm(y,ord=2)


# u = randn(N); v = randn(N)
# A = twist(u,v)
# D,V = eig(A)
# dispmat(D,nameof(D)); dispmat(V,nameof(V))


# print(dot(V[:,2],imag(V[:,0])))
# print(dot(V[:,2],real(V[:,0])))
# print(dot(real(V[:,0]),imag(V[:,0])))
# dot(V[:,-1],v)