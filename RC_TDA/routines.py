import os, dill; cwd = os.getcwd()+'/'
import numpy as np
from numpy import *
from numpy.random import *
from numpy.linalg import *
import numba as nb
from scipy import linalg 
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
layout = go.Layout(margin=go.layout.Margin(l=0,r=0,b=0,t=0))

def mkdir(cwdpath):
    if not os.path.exists(f"{cwd}{cwdpath}"):
        os.makedirs(f"{cwd}{cwdpath}")
    
def saveobj(obj,cwdpath):
    with open(f"{cwd}{cwdpath}",'wb') as f:
        dill.dump(obj,f)
    
def loadobj(cwdpath):
    with open(f"{cwd}{cwdpath}",'rb') as f:
        return dill.load(f)

def id(x): return x

@nb.njit
def create_Graph(n,p,rho):
    G = rand(n,n) < p
    G = (2*rand(n,n)-1)*np.triu(G,1)
    G = G + np.transpose(G)
    max_eigen = np.max(np.abs(np.linalg.eigvals(G)))
    G = G/max_eigen*rho
    return G

def angle(u,v):
    return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

def proj_S(x,*S_):
    d = len(S_)
    projS_x = np.zeros(len(x))
    for i in range(d):
        projS_x = projS_x + angle(x,S_[i])*np.linalg.norm(x)*S_[i]
    return projS_x


class ESN:
    def __init__(self, N,p,rho,leak,bias,factor):
        self.N = N
        self.sigma = 0.5
        self.leak = leak
        self.p = p
        self.xi = bias
        self.rho = rho
        self.factor = factor
        self.A = create_Graph(N,p,rho)
        self.stabilizing_method = False
        self.stabilizer_args = False
      
      
    def update(self,y=False):
        if y is False:
            self.state = (1-self.leak)*self.state + \
            self.leak*tanh(self.A@self.state + self.bias)
        else:
            self.state = (1-self.leak)*self.state + \
            self.leak*tanh(self.A@self.state + self.bias + self.factor*self.Win@y)
        
    
    def record(self, data,PreLen,InitLen,TrainLen):
        self.data = data
        self.InitLen = InitLen 
        self.TrainLen = TrainLen
        self.target_train = data[InitLen:InitLen+TrainLen].T
        self.bias = self.xi*ones(self.N)
        
        if len(data.shape) > 1:
            self.input_size = data.shape[-1]
        else:
            self.input_size = 1
        self.Win = self.sigma*(-1 + 2*rand(self.N,self.input_size))
        self.state = self.sigma*(-1 + 2*rand(self.N))
        
        self.R_train = zeros((self.N,TrainLen))
        for i in range(PreLen): # starting reservoir
            self.update()
        for i in range(InitLen): # injecting train input 
            self.update(y=data[i])
        for i in range(TrainLen):
            self.R_train[:,i] = self.state # recording train output
            self.update(y=data[InitLen+i])  
    
    
    def optimize(self,method,*args):
        if method == 'pinv':
            self.Wout = self.target_train@pinv(self.R_train) # training readout
        if method == 'pinv_reg':
            reg = args
            self.Wout = self.target_train@self.R_train.T\
                @pinv(self.R_train@self.R_train.T + reg*eye(self.N,self.N))
            #Wout = target_train*reservoir_train' * pinv(reservoir_train*reservoir_train' + reg*eye(N));
        if method == 'pinv_degrad':
            ratio = args
            split_pts = rint(self.R_train.shape[1]*array(ratio)).astype(int)
            Rs = split(self.R_train,split_pts,axis=1)
            Ys = split(self.target_train,split_pts,axis=1)
            Wout = zeros((self.N,self.input_size))
            for i in range(len(ratio)+1):
                Wout = Wout + pinv(Rs[i])@Ys[i]
            self.Wout = Wout/(len(ratio)+1)

        self.U, self.S, self.V = np.linalg.svd(self.A + self.Win@self.Wout)
        self.V = self.V.T
        self.blackholes = [self.V[:,i] for i in range(self.input_size)]
    
    
    def get_train_error(self,disp=False):
        self.train_err = norm(self.Wout@self.R_train-self.target_train)
        if disp is True:
            print(f'\tTrain Error: {self.train_err:.6f}')
        
        
    def stabilize(self,method,arg):
        self.stabilizing_method = method
        self.stabilizer_args = arg
      
      
    def generate_out(self,TestLen,
                     record_state=False,record_feedback_ratio=False,record_proj_ratio=False):
        self.TestLen = TestLen
        self.target = self.data[self.InitLen+self.TrainLen:self.InitLen+self.TrainLen+self.TestLen].T
        self.out = zeros((self.input_size,TestLen))
       
        if record_state is True:
            self.state_ = zeros((self.N,TestLen))
        if record_feedback_ratio is True:
            elf.feedback_ratio_ = zeros(TestLen)
        if record_proj_ratio is True:
            self.proj_ratio_ = zeros(TestLen)
        
        for i in range(TestLen): # testing prediction
            if self.stabilizing_method == 1:
                self.state = self.state - proj_S(self.state,*self.blackholes)
            if self.stabilizing_method == 2:
                self.state = self.state - self.stabilizer_args*proj_S(self.state,*self.blackholes)
            if self.stabilizing_method == 3:
                if norm(proj_S(self.state,*self.blackholes)) > self.stabilizer_args[0]:
                    self.state = self.state - self.stabilizer_args[1]*proj_S(self.state,*self.blackholes)
              
            self.out[:,i] = self.Wout@self.state
          
          if record_feedback_ratio is True:
              self.feedback_ratio_[i] = norm(self.Win@self.out[:,i])/norm(self.state)
            
          self.update(y=self.out[:,i])
          
          if record_state is True:
              self.state_[:,i] = self.state
          if record_proj_ratio is True:
              self.proj_ratio_[i] = norm(proj_S(self.state,*self.blackholes))/norm(self.state)
          
          
    def get_test_error(self,disp):
        self.test_err = norm(self.out-self.target, ord=2)
        if self.test_err > 1000:
            self.prediction_status = 0
        else:
            self.prediction_status = 1
        if disp is True:
            print(f'\tTest Error: {self.test_err:.6f}, status: {self.prediction_status}')
    
    
    def render_out(self):
        if self.input_size == 3:
            fig1 = go.Figure(go.Scatter3d(x=self.out[0,:],y=self.out[1,:],z=self.out[2,:],mode='lines'),
                    layout=layout)
        elif self.input_size == 2:
            fig1 = go.Figure(go.Scatter(x=self.out[0,:],y=self.out[1,:]),
                    layout=layout)
        else: fig1 = go.Figure()
        fig1.update_xaxes(range=[np.min(self.target), np.max(self.target)])
        fig1.update_yaxes(range=[np.min(self.target), np.max(self.target)])
          
        fig2 = make_subplots(self.input_size,1, horizontal_spacing=0)
        X = list(range(self.TestLen))
        for i in range(self.input_size):
            fig2.add_trace(go.Scatter(x=X,y=self.out[i,:]),row=i+1,col=1)
            fig2.append_trace(go.Scatter(x=X,y=self.target[i,:]),row=i+1,col=1)
        fig2.update_yaxes(range=[np.min(self.target), np.max(self.target)])

        return fig1, fig2
    
    
    def plot_singular_values(self):
        fig,ax = plt.subplots()
        x = arange(0,self.input_size)
        ax.bar(x,abs(self.S[:self.input_size]))
        ax.set_title(f'{self.input_size} Largest singular values of W')
        return fig
      
      
    def check_blackhole(self,n_eig_space):
        _, AW_eigvecs = eig(self.A+self.Win@self.Wout)
        vecs = AW_eigvecs[:,:n_eig_space]
        self.danger_ = zeros((self.TestLen,n_eig_space))
        for i in range(self.TestLen):
            for j in range(n_eig_space):
                self.danger_[i,j] = dot(self.state_[i,:],vecs[:,j])/\
                (norm(self.state_[:,i],ord=2)*norm(vecs[:,j],ord=2))


@nb.njit
def pointwise_distance(X,Y):
    n = X.shape[0]
    dist = 0
    for i in range(n):
      dist = dist + np.linalg.norm(X[i,:]-Y[i,:])
    return dist/n

  
@nb.njit
def rossler(xyz,_, a,b,c): 
    x = xyz[0]; y = xyz[1]; z = xyz[2]
    dxdt = -y-z
    dydt = x+a*y
    dzdt = b+z*(x-c)
    sol = np.array([dxdt,dydt,dzdt])
    return sol

  
@nb.njit
def RK4(F,iniy,tfin,h,*args):
    t = np.arange(0,tfin+0.1*h,h)
    soln = np.zeros((len(t),len(iniy)))
    
    soln[0,:] = iniy
    for i in range(len(t)-1):
      k1 = h * F(soln[i,:],t[i], *args)
      k2 = h * F(soln[i,:]+k1/2,t[i]+h/2, *args)
      k3 = h * F(soln[i,:]+k2/2,t[i]+h/2, *args)
      k4 = h * F(soln[i,:]+k3,t[i]+h, *args)
      soln[i+1,:] = soln[i,:]+(k1+2.0*k2+2.0*k3+k4)/6.0
    return t, soln


@nb.njit
def RK4_bifur(F,iniy,tfin,h, param,*args):
    t = np.arange(0,tfin+0.1*h,h)
    soln = np.zeros((len(t),len(iniy)))

    soln[0,:] = iniy
    for i in range(len(t)-1):
      k1 = h * F(soln[i,:],t[i], *args,param[i])
      k2 = h * F(soln[i,:]+k1/2,t[i]+h/2, *args,param[i])
      k3 = h * F(soln[i,:]+k2/2,t[i]+h/2, *args,param[i])
      k4 = h * F(soln[i,:]+k3,t[i]+h, *args,param[i])
      soln[i+1,:] = soln[i,:]+(k1+2.0*k2+2.0*k3+k4)/6.0
    return t, soln