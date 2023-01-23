from RPS_routines import Lattice, LatticeHistory, produce_params, mkdir, load_data, check_memory_usage
from joblib import Parallel, delayed
import numpy as np
import mgzip, dill
import matplotlib.pyplot as plt

# save_dir = '/home/hgyoon/Research/RPS/crit_mob2+_hists'; mkdir(save_dir)
# mobilities = 9*1e-4*np.ones(samples)

# save_dir = '/home/hgyoon/Research/RPS/crit_mob2-_hists'; mkdir(save_dir)
# mobilities = 3*1e-4*np.ones(samples)

# L = 200
# samples = 500
# params = produce_params(mobilities=mobilities,rho=1,k=1,L=L)

# def generate_sample(id):
#     print(f'Now generating sample {id} ...')
#     lattice = Lattice(X=np.random.randint(1,4,(L,L)),params=params[id])
#     init_hist, last_hist = lattice.evolve(
#         T=5*lattice.N,
#         modify_self=True,
#         save_history=True, 
#         save_interval=50,      
#         init_history_length=200,
#         last_history_length=100,
#         show_evolution=False
#     )
#     return (init_hist, last_hist, lattice.status, lattice.generation)
            
# hists = Parallel(n_jobs=24) (
#     delayed( generate_sample )(id) for id in range(samples)
# )

# hists = np.array(hists)
# init_hists, last_hists, stats, generations = \
#     hists[:,0].tolist(), hists[:,1].tolist(), hists[:,2], hists[:,3]

# with mgzip.open(f'{save_dir}/data.gz','wb') as f: dill.dump((init_hists,last_hists),f)
# np.savetxt(f'{save_dir}/L200_stats.dat',np.array((stats,generations)),fmt='%d')
# np.savetxt(f'{save_dir}/L200_mobilities.dat',mobilities,fmt='%d')


#----------------------------------------------------------------------------
#data_dir = '/home/hgyoon/Research/RPS/crit_mob2_hists-'
data_dir = '/home/hgyoon/Research/RPS/crit_mob2_hists+'

init_hists, last_hists = load_data(f'{data_dir}/data.gz')
mobilities = np.loadtxt(f'{data_dir}/L200_mobilities.dat')
summary = np.loadtxt(f'{data_dir}/L200_stats.dat')
params = produce_params(mobilities,rho=1,k=1,L=200)

last_gens = summary[1,:].astype(int)

fast_ids = np.where(last_gens < 10000)[0]
sudden_ids = np.where( (last_gens >= 10000) & (last_gens < 200000) )[0]
non_ids = np.where(last_gens == 200000)[0]

fig, ax = plt.subplots()
fig.suptitle
plt.hist(last_gens[last_gens < 200000], bins=200)
plt.hist(last_gens[fast_ids],bins=100)
plt.hist(last_gens[sudden_ids],bins=200)

save_dir = f'{data_dir}/p_ext_figs'; mkdir(save_dir)
mkdir(f'{save_dir}/fast'); mkdir(f'{save_dir}/sudden'); mkdir(f'{save_dir}/non')

for i in range(30):
    sample_id = fast_ids[i]
    print(f'\n\nNow for sample {sample_id} ...')
    hist:LatticeHistory = init_hists[sample_id]
    hist.p_exts(si=0,fi=30,T=10000).plot(ylim=[0,1],
                                         title=f'Fast_ext, Sample: {sample_id}',
                                         save_path=f'{save_dir}/fast/sample{sample_id}.png')
    
    try:
        sample_id = sudden_ids[i]
        print(f'\n\nNow for sample {sample_id} ...')
        hist:LatticeHistory = init_hists[sample_id]
        hist.p_exts(si=0,fi=30,T=10000).plot(ylim=[0,1],
                                            title=f'Sudden_ext, Sample: {sample_id}',
                                            save_path=f'{save_dir}/sudden/sample{sample_id}.png')
    except: pass
    
    sample_id = non_ids[i]
    print(f'\n\nNow for sample {sample_id} ...')
    hist:LatticeHistory = init_hists[sample_id]
    hist.p_exts(si=0,fi=30,T=10000).plot(ylim=[0,1],
                                         title=f'Non_ext, Sample: {sample_id}',
                                         save_path=f'{save_dir}/non/sample{sample_id}.png')

