import os, time
import numpy as np
from numpy.random import uniform, randint
from joblib import Parallel, delayed
from RPS_routines import EvolvingLattice, produce_prob_params, mkdir

data_dir = '/home/hgyoon/Research/RPS/crit_mob2'
mkdir(data_dir)
    
L = 200; N = L**2
num_samples = 2000
mobilities = uniform(6*1e-4,6*1e-4,num_samples)
np.savetxt(f'{data_dir}/L{L}_mobilities.dat',mobilities,delimiter=' ')

params = produce_prob_params(mobilities,rho=1,k=1,L=L)

def generate_sample(sample_id):
    st = time.time()
    status = 0
    lattice = EvolvingLattice(L, *params[sample_id])
    init_hist, last_hist, status, end_gen =\
        lattice.evolve(randint(1,4,(L,L)).astype(np.int8), T=int(5*N))
        
    np.savez_compressed(f'{data_dir}/L{L}_sample{sample_id}_hist',
                        init_hist=init_hist.astype(np.int8),
                        last_hist=last_hist.astype(np.int8))
    ft = time.time()
        
    return status, end_gen

final_stats = np.array( Parallel(n_jobs=24)\
    ( delayed(generate_sample)(sample_id) for sample_id in range(num_samples) ) )

np.savetxt(f'{data_dir}/L{L}_status.dat',final_stats.astype(int),
           delimiter=' ',fmt='%1d')

# hists = np.load(f'{data_dir}/L{L}_sample0_hist.npz')
# init_hist, last_hist = hists['init_hist'], hists['last_hist']

