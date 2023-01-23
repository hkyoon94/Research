import numpy as np
from RPS_routines import mkdir, produce_prob_params, draw_snapshots_
from joblib import Parallel, delayed

def load_samples(ids,key):
    def load_sample(id):
        return np.load(f'{data_dir}/L200_sample{id}_hist.npz')[key]
    return Parallel(n_jobs=24)( delayed(load_sample)(id) for id in ids )


data_dir = '/home/hgyoon/Research/RPS/crit_mob2'
L = 200; N = L**2
mobilities = np.loadtxt(f'{data_dir}/L200_mobilities.dat')
stats = np.loadtxt(f'{data_dir}/L200_status.dat').astype(int)
status, times = stats[:,0], stats[:,1]
prob_params = produce_prob_params(mobilities,rho=1,k=1,L=L)

init_hists = load_samples(range(2000),'init_hist')
last_hists = load_samples(range(2000),'last_hist')


def draw_category(category_ids, save_dir, draw_init=True):
    mkdir(save_dir)
    
    for id in category_ids:
        if draw_init is True:
            draw_snapshots_(init_hists[id],suptitle=f'Initial {len(init_hists[id])} snapshots')\
                .savefig(f'{save_dir}/sample{id}_init.png')
        if status[id] == 1: last_suptitle = f'Extinction at t={times[id]}'
        else: last_suptitle = f'Non-extinct till t=200000'
        draw_snapshots_(last_hists[id],suptitle=last_suptitle)\
            .savefig(f'{save_dir}/sample{id}_last.png')

draw_category( # fast extincts
    category_ids = np.where(times < 10000)[0],
    save_dir = '/home/hgyoon/Research/RPS/crit_mob2_snapshots/fast_ext',
    draw_init = False
)

draw_category( # sudden extincts
    category_ids = np.where((times >= 10000)&(times < 200000))[0],
    save_dir = '/home/hgyoon/Research/RPS/crit_mob2_snapshots/sudden_ext'
)

draw_category( # non-extincts
    category_ids = np.where(times == 200000)[0][:200],
    save_dir = '/home/hgyoon/Research/RPS/crit_mob2_snapshots/non_ext'
)