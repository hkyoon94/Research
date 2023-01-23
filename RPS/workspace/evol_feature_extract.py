import os, time
import numpy as np
from numba import njit
from RPS_routines import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def load_samples(ids,key):
    def load_sample(id):
        return np.load(f'{data_dir}/L200_sample{id}_hist.npz')[key]
    
    return Parallel(n_jobs=24)( delayed(load_sample)(id) for id in ids )

data_dir = '/home/hgyoon/Research/RPS/crit_mob2'
species_nums_dir = '/home/hgyoon/Research/RPS/crit_mob_species_dens'
mkdir(species_nums_dir)
L = 200; N = L**2

mobilities = np.loadtxt(f'{data_dir}/L200_mobilities.dat')
stats = np.loadtxt(f'{data_dir}/L200_status.dat').astype(int)
prob_params = produce_params(mobilities,rho=1,k=1,L=L)

status, times = stats[:,0], stats[:,1]
critical_ids = np.where( (times > 10000) & (times < 200000) )[0]
num_crit = len(critical_ids)
status, times = status[critical_ids], times[critical_ids]

fig, ax = plt.subplots()
ax.hist(times,bins=100)
plt.show()

last_Hist = load_samples(critical_ids,'last_hist')
init_Hist = load_samples(critical_ids,'init_hist')


# feature 1: density ---------------------------------------------------------------------------------#
@njit
def count_species_hist_(hist):
    hist_len = hist.shape[0]
    species_nums = np.zeros((hist_len,4))
    
    for i in range(hist_len):
        species_nums[i] = _count_species(hist[i])
        
    return species_nums

@njit
def count_species_Hist_(Hist):
    Hist_filtered_dens = []
    samples = len(Hist)
    for id in range(samples):
        Hist_filtered_dens.append(count_species_hist_(Hist[id])/N)
    
    return Hist_filtered_dens

st = time.time()
Dens = count_species_Hist_(last_Hist,num_crit)
ft = time.time()
print(f'Elapsed time: {ft-st:6f}')

