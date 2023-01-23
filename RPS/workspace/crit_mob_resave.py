import os, time, dill, pickle, lzma, gzip, bz2, zlib, mgzip
import numpy as np
from RPS_routines import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

data_dir = '/home/hgyoon/Research/RPS/crit_mob2'
L = 200; N = L**2

working_num = 2000
st = time.time()
init_hists = load_samples(data_dir, range(working_num), 'init_hist')
last_hists = load_samples(data_dir, range(working_num), 'last_hist')
ft = time.time(); print(f'{ft-st}')

mobilities = np.loadtxt(f'{data_dir}/L200_mobilities.dat')[:working_num]
stats = np.loadtxt(f'{data_dir}/L200_status.dat').astype(int)[:working_num]

save_dir = '/home/hgyoon/Research/RPS/crit_mob3'
mkdir(save_dir)

init_gens = []
last_gens = []

for sample_id in range(working_num):
    init_hist = init_hists[sample_id]
    last_hist = last_hists[sample_id]
    
    init_len = len(init_hist); last_len = len(last_hist)
    fin_time = np.loadtxt(f'{data_dir}/L200_status.dat').astype(int)[sample_id][1]

    if len(init_hist)<200:
        init_gen = np.linspace(50,50*init_len,init_len).astype(int)
        last_gen = init_gen.copy()
    else:
        init_gen = np.linspace(50,10000,200).astype(int)
        last_gen = np.linspace(fin_time-199*50,fin_time,200).astype(int)
        
    init_gens.append(init_gen)
    last_gens.append(last_gen)
    
init_hists2 = []
last_hists2 = []

for sample_id in range(working_num):
    
    init_hist = init_hists[sample_id]; init_gen = init_gens[sample_id]
    last_hist = last_hists[sample_id]; last_gen = last_gens[sample_id]
    
    init_hist2 = []
    last_hist2 = []
    
    for i in range(len(init_hist)):
        init_hist2.append( Lattice(X=init_hist[i], generation=init_gen[i]) )
    for i in range(len(last_hist)):
        last_hist2.append( Lattice(X=last_hist[i], generation=last_gen[i]) )
        
    init_hists2.append(init_hist2)
    last_hists2.append(last_hist2)

    

st = time.time()
for sample_id in range(working_num):
    data = (init_hists2[sample_id], last_hists2[sample_id])
    with mgzip.open(f'{save_dir}/sample{sample_id}_hist.gz', 'wb', thread=24) as f: 
        dill.dump(data,f)
    print(f'processed sample {sample_id}')
ft = time.time()



st = time.time()
data = (init_hists2, last_hists2)
with mgzip.open(f'{save_dir}/samples.gz', 'wb', thread=24) as f: 
    dill.dump(data,f)
ft = time.time()

st = time.time()
with mgzip.open(f'{save_dir}/samples.gz', 'rb', thread=24) as f: 
    Init_hists, Last_hists = dill.load(f)
ft = time.time(); print(f'{ft-st}')



#-----------------------------------------
# wrapping raw .npz data to class
st = time.time()
init_hist2 = [] 
for i in range(len(init_hist)):
    init_hist2.append(Lattice(X=init_hist[i],generation=init_gens[i]))
    
last_hist2 = []
for i in range(len(init_hist)):
    last_hist2.append(Lattice(X=init_hist[i],generation=last_gens[i]))
ft = time.time()
print(f'{ft-st}')
    
#-----------------------------------------
# savez import and load (filesize 2.7MB)
st = time.time()
np.savez_compressed(f'{save_dir}/savez',init_hist=init_hist,last_hist=last_hist,
                    init_gens=init_gens,last_gens=last_gens)
ft = time.time()
print(f'{ft-st}')

st = time.time()
a = np.load(f'{save_dir}/savez.npz')
ft = time.time()
print(f'{ft-st}')

#-----------------------------------------
# dill read and load (filesize 16MB)
st = time.time()
with open(f'{save_dir}/dill.pkl','wb') as f:
    dill.dump((init_hist2,last_hist2),f)
ft = time.time()
print(f'{ft-st}')

st = time.time()
with open(f'{save_dir}/dill.pkl','rb') as f:
    init_hist3, last_hist3 = dill.load(f)
ft = time.time()
print(f'{ft-st}')

#-----------------------------------------
# lzma-pickle (filesize 1.2MB)
st = time.time()
with lzma.open(f'{save_dir}/lzma_pkl.pkl','wb') as f:
    dill.dump((init_hist2,last_hist2),f)
ft = time.time()
print(f'{ft-st}')

st = time.time()
with lzma.open(f'{save_dir}/lzma_pkl.pkl','rb') as f:
    init_hist4, last_hist4 = pickle.load(f)
ft = time.time()
print(f'{ft-st}')

#-----------------------------------------
# lzma-dill (filesize 1.2MB)
st = time.time()
with lzma.open(f'{save_dir}/lzma_dill.xz','wb') as f:
    dill.dump((init_hist2,last_hist2),f)
ft = time.time()
print(f'{ft-st}')

st = time.time()
with lzma.open(f'{save_dir}/lzma_dill.xz','rb') as f:
    init_hist5, last_hist5 = dill.load(f)
ft = time.time()
print(f'{ft-st}')

#-----------------------------------------
# gzip-dill (filesize 2.6MB)
st = time.time()
with gzip.open(f'{save_dir}/gzip_dill.pkl','wb') as f:
    dill.dump((init_hist2,last_hist2),f)
ft = time.time()
print(f'{ft-st}')

st = time.time()
with gzip.open(f'{save_dir}/gzip_dill.pkl','rb') as f:
    init_hist6, last_hist6 = dill.load(f)
ft = time.time()
print(f'{ft-st}')

#-----------------------------------------
# bz2-dill (filesize 2.4MB)
st = time.time()
with bz2.open(f'{save_dir}/bz2_dill.pkl','wb') as f:
    dill.dump((init_hist2,last_hist2),f)
ft = time.time()
print(f'{ft-st}')

st = time.time()
with bz2.open(f'{save_dir}/bz2_dill.pkl','rb') as f:
    init_hist7, last_hist7 = dill.load(f)
ft = time.time()
print(f'{ft-st}')

#-----------------------------------------
# mgzip-dill (filesize 2.6MB)
st = time.time()
with mgzip.open(f'{save_dir}/mgzip_dill.pkl','wb') as f:
    dill.dump((init_hist2,last_hist2),f)
ft = time.time()
print(f'{ft-st}')

st = time.time()
with mgzip.open(f'{save_dir}/mgzip_dill.pkl','rb') as f:
    init_hist8, last_hist8 = dill.load(f)
ft = time.time()
print(f'{ft-st}')