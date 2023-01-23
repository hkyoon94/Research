import os, psutil, dill, mgzip
from time import time
from copy import deepcopy
import numpy as np
from numba import njit
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
cmap_RPS = LinearSegmentedColormap.from_list(
    'RPS',[[0,0,0],[1,0,0],[0,0,1],[0.8,0.8,0]], N=4
)
color_RPS = ['r','b','y']
color_RPS_k = ['k','r','b','y']


def check_memory_usage():
    process = psutil.Process()
    print(f'Current session memory usage: {(process.memory_info().rss/1024**3):4f} GBs')
    

def mkdir(dir_path:str) -> None:
    if os.path.isdir(dir_path) is False: os.mkdir(dir_path)


def load_data(data_dir):
    st = time()
    with mgzip.open(f'{data_dir}','rb',thread=24) as f:
        Data = dill.load(f)
    ft = time()
    print(f'Sample loaded in {(ft-st):4f} seconds.')
    
    return Data


def produce_params(mobilities:np.ndarray, rho:float, k:float, L:int) -> np.ndarray:
    try: samples = len(mobilities)
    except:
        mobilities = [mobilities]; samples = 1
    prob_params = np.zeros((samples,3))
    
    for i, mobility in enumerate(mobilities):
        epsilon = 2*mobility*(L**2)
        normalizer = epsilon + rho + k
        prob_params[i,0] = epsilon/normalizer
        prob_params[i,1] = rho/normalizer
        prob_params[i,2] = k/normalizer
    
    return prob_params.squeeze()


def build_workspace(Data,params,summary): 
    init_Xss = Data['init_Xs']; init_genss = Data['init_gens']
    last_Xss = Data['last_Xs']; last_genss = Data['last_gens']
    num_samples = len(init_Xss)
    
    init_hists = []; last_hists = []
    
    st = time()
    for i in range(num_samples):
        init_Xs = init_Xss[i]; last_Xs = last_Xss[i]
        init_gens = init_genss[i]; last_gens = last_genss[i]
        
        init_hist = []; last_hist = []
        
        for j in range(len(init_Xs)): 
            init_hist.append(Lattice(init_Xs[j],params[i],init_gens[j]))
        for j in range(len(last_Xs)): 
            last_hist.append(Lattice(last_Xs[j],params[i],last_gens[j]))
        
        init_hists.append(LatticeHistory(init_hist,final_status=summary[i,0]))
        last_hists.append(LatticeHistory(last_hist,final_status=summary[i,0]))
    ft = time()
    
    print(f'Data built in {(ft-st):4f} seconds.\n')
    check_memory_usage()
    
    return np.array(init_hists), np.array(last_hists)



#-----------------------------------------------------------------------------------------------
class Lattice:
    def __init__(self, X:np.ndarray, params, generation=0):
        self.X = X
        self.L = X.shape[0]
        self.N = self.L**2
        self.params = params
        self.generation = generation
    
    @property
    def species_num(self): return _count_species(self.X)
        
    @property
    def status(self):
        if np.sum(self.species_num[1:] == 0) == 0: return 0
        else: return 1    
        
    def snapshot(self, ax:plt.Axes=None, inches=(5,5), cmap=cmap_RPS):
        if ax is not None: _snapshot_on_ax(X=self.X, ax=ax, cmap=cmap)
        else: 
            print(f'Generation: {self.generation}')
            fig, ax = plt.subplots()
            fig.set_size_inches(*inches)
            _snapshot_on_ax(X=self.X, ax=ax, cmap=cmap)
            plt.show()
            plt.close()
    
    def periodic_snapshot(self, ax:plt.Axes=None, repeat=3, inches=(7,7), cmap=cmap_RPS):
        def concatenate_n(X,axis):
            for i in range(repeat-1):
                if i == 0: X_ = np.concatenate((X,X),axis=axis)
                else: X_ = np.concatenate((X_,X),axis=axis)
            return X_
        
        print(f'Generation: {self.generation}')
        X_periodic = concatenate_n(concatenate_n(X=self.X,axis=0),1)
        if ax is not None: _snapshot_on_ax(X=X_periodic, ax=ax, cmap=cmap)
        else: 
            fig, ax = plt.subplots()
            fig.set_size_inches(*inches)
            _snapshot_on_ax(X=X_periodic, ax=ax, cmap=cmap)
            plt.show()
            plt.close()
    
    def evolve(self, T, modify_self=False,
               save_interval=50, draw_interval=None,
               init_history_length=200, last_history_length=200, save_history=True, 
               show_evolution=True, print_log=True):
            
        if not modify_self: self = deepcopy(self)
        draw_interval = save_interval*4 if draw_interval is None else draw_interval
        
        if save_history:
            init_hist, init_hist_ct = LatticeHistory(), 0 
            last_hist, last_hist_ct = LatticeHistory(), 0
            
            def _update_history():
                nonlocal init_hist_ct, last_hist_ct, init_hist, last_hist
                if init_hist_ct < init_history_length: 
                    init_hist.hist.append(deepcopy(self))
                if last_hist_ct < last_history_length: 
                    last_hist.hist.append(deepcopy(self))
                else: last_hist.hist.append(deepcopy(self)); last_hist.hist.pop(0)
                init_hist_ct += 1; last_hist_ct += 1
                
            _update_history()
        
        init_gen, end_gen = self.generation, self.generation + int(T)
        X, species_num, L = self.X, self.species_num, self.L
        
        st = time()
        for self.generation in range(init_gen+1, end_gen+1):
            X, species_num = _simulate_one_generation(
                X=X, species_num=species_num, L=L, 
                pe=self.params[0], pr=self.params[1], pk=self.params[2]
            )
            if save_history and self.generation % save_interval == 0: 
                _update_history()
            if show_evolution and self.generation % draw_interval == 0: 
                self.snapshot(inches=(1.5,1.5))
            if not np.sum(species_num[1:] == 0) == 0: break  
        ft = time()
        self.X = X
        
        if print_log: print(f'Evolution ended ({ft-st:6f} sec).\n\tStatus: {self.status}, \
            End generation: {self.generation}')
        if show_evolution: self.snapshot()
        
        if save_history:
            init_hist.final_status = self.status
            last_hist.final_status = self.status
            return init_hist, last_hist

    def measure_p_ext(self, ensemble=48, T=None):
        T = 5*self.N if T is None else T
    
        def run_trial():
            trial_lattice = deepcopy(self)
            trial_lattice.evolve(
                modify_self=True,
                T=T,
                save_history=False, show_evolution=False, print_log=False,
            )
            return trial_lattice.status, trial_lattice.generation
        
        st = time()
        ensemble_stats = np.array(
            Parallel(n_jobs=24)(delayed(run_trial)() for _ in range(ensemble))
        )
        ft = time()
        p_ext = np.sum(ensemble_stats[:,0])/ensemble
        print(f'p_ext: {p_ext:4f} ( Elapsed time: {ft-st:4f} sec )')
        
        return p_ext, ensemble_stats[:,1]
    
    # EXTRACTABLE SINGLE FEATURES
    def density(self): return self.species_num/self.N
    
    def score_1(self): return _score_1(self.X, self.L)
    
    def score_2(self): return _score_2(self.X, self.L)

def _snapshot_on_ax(X:np.ndarray, ax:plt.Axes, cmap=cmap_RPS):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(X, cmap=cmap)

# FAST ROUTINES 
@njit
def _simulate_one_generation(X:np.ndarray, species_num:np.ndarray, 
                             L:int, pe:np.float64, pr:np.float64, pk:np.float64
                             ) -> tuple[np.ndarray, np.ndarray]:
    
    p = np.random.random(L**2); # probability to determine an action
    R = np.random.randint(0,L,(L**2,2)) # lattice position for an action
    c = np.random.randint(0,4,L**2) # neighbor direction for current action

    #2x4 matrix indicating 4 directions: L[:,1]=right, L[:,2]=left, L[:,3]=up, L[:,4]=down
    drct = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    
    for i in range(L**2): 
        N1 = R[i,0] + drct[c[i],0] # N1 = row index of neighbor, N2 = column index of neighbor
        N2 = R[i,1] + drct[c[i],1]

        if N1 == L: N1 = 0
        elif N1 == -1: N1 = L-1
        if N2 == L: N2 = 0
        elif N2 == -1: N2 = L-1

        neighbor = X[N1,N2]
        host = X[R[i,0],R[i,1]]
        
        if p[i] < pe: # exchange motion
            if (host + neighbor) > 0:
                X[N1,N2] = host
                X[R[i,0],R[i,1]] = neighbor

        elif p[i] < pe + pr: # reproduction
            if (host * neighbor) == 0 and (host + neighbor) > 0:
                species_num[0] -= 1
                if neighbor == 0: # host reproduces at neighbor
                    species_num[host] += 1
                    X[N1,N2] = host
                else: # neighbor reproduces at host
                    species_num[neighbor] += 1
                    X[R[i,0],R[i,1]] = neighbor
                
        elif p[i] < pe + pr + pk: # predation
            if (host * neighbor) > 0 and not (host - neighbor) == 0:
                species_num[0] += 1
                if (host - neighbor) == -1 or (host - neighbor) == 2: # host predates neighbor
                    species_num[neighbor] -= 1
                    X[N1,N2] = 0
                elif (host - neighbor) == 1 or (host - neighbor) == -2: # neighbor predates host
                    species_num[host] -= 1
                    X[R[i,0],R[i,1]] = 0
                    
    return X, species_num

@njit
def _count_species(X:np.ndarray) -> np.ndarray:
    species_num = np.array([0,np.sum(X==1),np.sum(X==2),np.sum(X==3)]).astype(np.int32)
    species_num[0] = X.shape[0]**2 - np.sum(species_num[1:])
    
    return species_num

@njit
def _score_1(X:np.ndarray, L:int) -> np.ndarray:
    x = np.zeros((L+1,L+1))
    x[:-1,:-1] = X; x[-1,:-1] = X[0,:]; x[:-1,-1] = X[:,0] # padding periodic bdry
    ct1 = np.zeros(3) # vulnerability
    
    for i in range(L):
        for j in range(L):
            host = x[i,j]
            down = x[i,j+1]; right = x[i+1,j]
            if host == 1: 
                if down == 2: ct1[0] += 1
                if right == 2: ct1[0] += 1
            elif host == 2:
                if down == 3: ct1[1] += 1
                if right == 3: ct1[1] += 1
            elif host == 3:
                if down == 1: ct1[2] += 1
                if right == 1: ct1[2] += 1
   
    u = np.zeros(3)
    for i in range(3):
        n = np.sum(X==(i+1))
        u[i] = 1 if n == 0 else ct1[i]/(2*n)
        
    return u

@njit
def _score_2(X:np.ndarray, L:int) -> np.ndarray:
    x = np.zeros((L+1,L+1))
    x[:-1,:-1] = X; x[-1,:-1] = X[0,:]; x[:-1,-1] = X[:,0] # padding periodic bdry
    
    ct1 = np.zeros(3) # vulnerability
    ct2 = np.zeros(3) # superiority
    
    for i in range(L):
        for j in range(L):
            host = x[i,j]
            down = x[i,j+1]; right = x[i+1,j]
            if host == 1: 
                if down == 2: ct1[0] += 1
                elif down == 3: ct2[0] += 1
                if right == 2: ct1[0] += 1
                elif right == 3: ct2[0] += 1
            elif host == 2:
                if down == 3: ct1[1] += 1
                elif down == 1: ct2[1] += 1
                if right == 3: ct1[1] += 1
                elif right == 1: ct2[1] += 1
            elif host == 3:
                if down == 1: ct1[2] += 1
                elif down == 2: ct2[2] += 1
                if right == 1: ct1[2] += 1
                elif right == 2: ct2[2] += 1
   
    u = np.zeros(3)
    for i in range(3):
        n = np.sum(X==(i+1))
        u[i] = 0 if n == 0 else (ct2[i]-ct1[i])*n
        
    return u



#------------------------------------------------------------------------------------------------
class LatticeHistory:
    def __init__(self, hist:list[Lattice]=None, final_status=0):
        self.hist = [] if hist is None else hist
        self.final_status = final_status
        
    @property 
    def length(self): return len(self.hist)
    
    @property
    def Xs(self): return np.array([lattice.X for lattice in self.hist])
    
    @property
    def generations(self): return np.array([lattice.generation for lattice in self.hist])
    
    def snapshots(self, suptitle=None, col_num=10, inches=2, cmap=cmap_RPS):
        num_plots = self.length
        row_num, r = divmod(num_plots,col_num)
        if r != 0: row_num+=1
        plot_position = range(1,num_plots+1)
        
        fig = plt.figure()
        fig.set_size_inches(col_num*inches,row_num*inches)
        fig.suptitle(suptitle, fontsize=15)
        plt.margins(tight=True)
        plt.axis('off')
        
        for i in range(num_plots):
            ax = fig.add_subplot(row_num,col_num,plot_position[i])
            _snapshot_on_ax(X=self.hist[i].X, ax=ax, cmap=cmap)
            ax.set_title(self.hist[i].generation)
            
        plt.close()
        
        return fig
    
    # EXTRACTABLE TIME-SERIES FEATURES
    def densities(self,si=0,fi=None):
        return TimeSeries(x = self.generations[si:fi],
                          y = [lattice.density() for lattice in self.hist[si:fi]],
                          x_label = 'Generations', y_label = 'Density',
                          plot_color = color_RPS_k)
        
    def p_exts(self,si=0,fi=None,**kwargs): 
        return TimeSeries(x = self.generations[si:fi],
                          y = [lattice.measure_p_ext(**kwargs)[0] for lattice in self.hist[si:fi]],
                          x_label = 'Generations', y_label = 'P_ext',
                          plot_color = 'k')
        
    def score_1(self,si=0,fi=None): 
        return TimeSeries(x = self.generations[si:fi],
                          y = [lattice.score_1() for lattice in self.hist[si:fi]],
                          x_label = 'Generations', y_label = 'Score 1',
                          plot_color = color_RPS)

    def score_2(self,si=0,fi=None):
        return TimeSeries(x = self.generations[si:fi],
                          y = [lattice.score_2() for lattice in self.hist[si:fi]],
                          x_label = 'Generations', y_label = 'Score 1',
                          plot_color = color_RPS)
    
@njit
def _score_time_derivative_norm(sample_scores:np.ndarray) -> np.ndarray:
    l = len(sample_scores)
    norms = np.zeros(l-1)
    for i in range(0,l):
        norms[i] = np.linalg.norm(sample_scores[i+1]-sample_scores[i])
        
    return norms


class TimeSeries:
    def __init__(self, x:list, y:list, x_label=None, y_label=None, plot_color=None):
        self.x = x
        self.y = y
        self.x_label = x_label
        self.y_label = y_label
        self.plot_color = plot_color
    
    def plot(self, title=None, xlim=None, ylim=None, save_path=None):
        fig, ax = plt.subplots()
        y = np.array(self.y)
        try:
            for i in range(y.shape[1]): ax.plot(self.x,y[:,i],color=self.plot_color[i])
        except: ax.plot(self.x, y, '-o',color=self.plot_color)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if title is not None: ax.set_title(title)
        ax.grid(True)
        plt.close()
        
        if save_path is None: return fig
        else: fig.savefig(save_path)
        

        
#-----------------------------------------------------------------------------------------------
class ListofLatticeHistory:
    def __init__(self, hists:list[LatticeHistory]):
        self.hists = hists
        self.num_hists = len(self.hists)
        
    def list_density(self,si=0,fi=None): 
        return ListofTimeSeries([hist.densities(si,fi) for hist in self.hists])
    
    def list_p_exts(self,si=0,fi=None,**kwargs):
        return ListofTimeSeries([hist.p_exts(si,fi,**kwargs) for hist in self.hists])
    
    def list_score_1(self,si=0,fi=None):
        return ListofTimeSeries([hist.score_1(si,fi) for hist in self.hists])
    
    def list_score_2(self,si=0,fi=None):
        return ListofTimeSeries([hist.score_2(si,fi) for hist in self.hists])
    

class ListofTimeSeries:
    def __init__(self, xs:list[TimeSeries]):
        self.xs = xs
        self.num_series = len(self.xs)
    
    def plots(self, xlim=None, ylim=None, inches=(5,3), save_path=None, **kwargs):
        fig, axes = plt.subplots(self.num_series,1); ax:list[plt.Axes]
        fig.set_size_inches(inches[0], self.num_series*inches[1])
        for id in range(self.num_series):
            series = self.xs[id]
            x = series.x
            y = np.array(series.y)
            ax:plt.Axes = axes[id]
            try: 
                for i in range(y.shape[1]): ax.plot(x,y[:,i],color=series.plot_color[i])
            except: ax.plot(x,y,'-o',color=series.plot_color)
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_xlabel(series.x_label,fontsize=7)
            ax.set_ylabel(series.y_label,fontsize=7)
            ax.grid(True)
            
        plt.margins(tight=True)
        plt.close()
        
        if save_path is None: return fig
        else: fig.savefig(save_path)
        