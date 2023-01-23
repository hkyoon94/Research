import os, importlib, pytictoc; timer=pytictoc.TicToc()
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

import RPS_routines; importlib.reload(RPS_routines)
from RPS_routines import *
check_memory_usage()

Data = load_data('/home/hgyoon/Research/RPS/crit_mob_hists/data.gz')
mobilities = np.loadtxt('/home/hgyoon/Research/RPS/crit_mob_hists/L200_mobilities.dat')
summary = np.loadtxt('/home/hgyoon/Research/RPS/crit_mob_hists/L200_status.dat')
params = produce_params(mobilities,rho=1,k=1,L=200)


#------------------------------------------------------------------------------------

init_hists:list[LatticeHistory]
last_hists:list[LatticeHistory]
init_hists, last_hists = build_workspace(Data, params, summary)

last_gens = summary[:,1].astype(int)

fast_ids = np.where(last_gens < 10000)[0]
sudden_ids = np.where( (last_gens > 10000) & (last_gens < 200000) )[0]
non_ids = np.where(last_gens == 200000)[0]


hist = init_hists[fast_ids[0]]
hist.densities().plot(ylim=[0,1],xlim=[0,1000])

hist = init_hists[non_ids[0]]
hist.densities().plot(ylim=[0,1],xlim=[0,1000])

#---------------------------------------------------------------------------------------------------
# save_dir = '    

# init_hists_fast.list_density().plots(xlim=[0,10000],ylim=[-0.05,1.05],
#                                      save_path=f'{save_dir}/fast_dens1.png')
# init_hists_sudden.list_density().plots(xlim=[0,10000],ylim=[-0.05,1.05],
#                                        save_path=f'{save_dir}/sudden_dens1.png')
# init_hists_non.list_density().plots(xlim=[0,10000],ylim=[-0.05,1.05],
#                                     save_path=f'{save_dir}/last_dens1.png')

# init_hists_fast.list_density().plots(xlim=[0,1500],ylim=[0.1,0.9],
#                                      save_path=f'{save_dir}/fast_dens2.png')
# init_hists_sudden.list_density().plots(xlim=[0,1500],ylim=[0.1,0.9],
#                                        save_path=f'{save_dir}/sudden_dens2.png')
# init_hists_non.list_density().plots(xlim=[0,1500],ylim=[0.1,0.9],
#                                     save_path=f'{save_dir}/last_dens2.png')

# init_hists_fast.list_score_1().plots(xlim=[0,10000],ylim=[-0.05,1.05],
#                                      save_path=f'{save_dir}/fast_score1.png')
# init_hists_sudden.list_score_1().plots(xlim=[0,10000],ylim=[-0.05,1.05],
#                                        save_path=f'{save_dir}/sudden_score1.png')
# init_hists_non.list_score_1().plots(xlim=[0,10000],ylim=[-0.05,1.05],
#                                     save_path=f'{save_dir}/last_score1.png')

# init_hists_fast.list_score_1().plots(xlim=[0,1500],ylim=[0,0.6],
#                                      save_path=f'{save_dir}/fast_score2.png')
# init_hists_sudden.list_score_1().plots(xlim=[0,1500],ylim=[0,0.6],
#                                        save_path=f'{save_dir}/sudden_score2.png')
# init_hists_non.list_score_1().plots(xlim=[0,1500],ylim=[0,0.6],
#                                     save_path=f'{save_dir}/last_score2.png')

#---------------------------------------------------------------------------------------------------
# save_dir = '/home/hgyoon/Research/RPS/figs2'

# for i in range(50):
    # sample_id = fast_ids[i]
    # print(f'\n\nNow for sample {sample_id} ...')
    # hist:LatticeHistory = init_hists[sample_id]
    # hist.p_exts(si=0,fi=30,T=10000).plot(ylim=[0,1],
    #                                      title=f'Fast_ext, Sample: {sample_id}',
    #                                      save_path=f'{save_dir}/fast_sample{sample_id}.png')
    
    # sample_id = sudden_ids[i]
    # print(f'\n\nNow for sample {sample_id} ...')
    # hist:LatticeHistory = init_hists[sample_id]
    # hist.p_exts(si=0,fi=30,T=10000).plot(ylim=[0,1],
    #                                      title=f'Sudden_ext, Sample: {sample_id}',
    #                                      save_path=f'{save_dir}/sudden_sample{sample_id}.png')
    
    # sample_id = non_ids[i]
    # print(f'\n\nNow for sample {sample_id} ...')
    # hist:LatticeHistory = init_hists[sample_id]
    # hist.p_exts(si=0,fi=30,T=10000).plot(ylim=[0,1],
    #                                      title=f'Non_ext, Sample: {sample_id}',
    #                                      save_path=f'{save_dir}/non_sample{sample_id}.png')


#----------------------------------------------------------------------------------------------------
save_dir = '/home/hgyoon/Research/RPS/figs3'
mkdir(save_dir); mkdir(f'{save_dir}/fast'); mkdir(f'{save_dir}/sudden'); mkdir(f'{save_dir}/non')

for i in range(50):
    sample_id = fast_ids[i]
    hist:LatticeHistory = init_hists[sample_id]
    hist.densities().plot(ylim=[0,1],
                          xlim=[0,1500],
                          title=f'Fast_ext, Sample: {sample_id}',
                          save_path=f'{save_dir}/fast/sample{sample_id}.png')
    
    sample_id = sudden_ids[i]
    hist:LatticeHistory = init_hists[sample_id]
    hist.densities().plot(ylim=[0,1],
                          xlim=[0,1500],
                          title=f'Sudden_ext, Sample: {sample_id}',
                          save_path=f'{save_dir}/sudden/sample{sample_id}.png')
    
    sample_id = non_ids[i]
    hist:LatticeHistory = init_hists[sample_id]
    hist.densities().plot(ylim=[0,1],
                          xlim=[0,1500],
                          title=f'Non_ext, Sample: {sample_id}',
                          save_path=f'{save_dir}/non/sample{sample_id}.png')
    
    
save_dir = '/home/hgyoon/Research/RPS/figs4'
mkdir(save_dir); mkdir(f'{save_dir}/fast'); mkdir(f'{save_dir}/sudden'); mkdir(f'{save_dir}/non')

for i in range(50):
    sample_id = fast_ids[i]
    hist:LatticeHistory = init_hists[sample_id]
    hist.score_1().plot(ylim=[0,1],
                          xlim=[0,1500],
                          title=f'Fast_ext, Sample: {sample_id}',
                          save_path=f'{save_dir}/fast/sample{sample_id}.png')
    
    sample_id = sudden_ids[i]
    hist:LatticeHistory = init_hists[sample_id]
    hist.score_1().plot(ylim=[0,1],
                          xlim=[0,1500],
                          title=f'Sudden_ext, Sample: {sample_id}',
                          save_path=f'{save_dir}/sudden/sample{sample_id}.png')
    
    sample_id = non_ids[i]
    hist:LatticeHistory = init_hists[sample_id]
    hist.score_1().plot(ylim=[0,1],
                          xlim=[0,1500],
                          title=f'Non_ext, Sample: {sample_id}',
                          save_path=f'{save_dir}/non/sample{sample_id}.png')
    
    
    
#------------------------------------------------------------------------------------------------------


