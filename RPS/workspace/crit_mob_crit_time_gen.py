import dill
import numpy as np
from RPS_routines import EvolvingLattice, load_samples, mkdir, produce_prob_params
#import matplotlib.pyplot as plt

data_dir = '/home/hgyoon/Research/RPS/crit_mob2'

init_hists = load_samples(data_dir, range(2000), 'init_hist')
last_hists = load_samples(data_dir, range(2000), 'last_hist')
mobilities = np.loadtxt(f'{data_dir}/L200_mobilities.dat')
stats = np.loadtxt(f'{data_dir}/L200_status.dat').astype(int)

L = 200; N = L**2
status, times = stats[:,0], stats[:,1]
prob_params = produce_prob_params(mobilities,rho=1,k=1,L=L)[0]

save_dir1 = '/home/hgyoon/Research/RPS/crit_mob2_crit_time/fast_ext'
ids1 = np.where(times < 10000)[0][:100]

save_dir2 = '/home/hgyoon/Research/RPS/crit_mob2_crit_time/sudden_ext'
ids2 = np.where((times > 10000)&(times < 200000))[0][:]

save_dir3 = '/home/hgyoon/Research/RPS/crit_mob2_crit_time/non_ext'
ids3 = np.where(times == 200000)[0][:100]


start_from_end = 101 # 101
measure_intervals = 80 # 70
ensemble_num = 50 # 50
fin_gen = 1*N # 1*N


def generate_samples_hist(ids, save_dir, hists, init_or_last):
    mkdir(save_dir)
    
    for iteration, sample_id in enumerate(ids):
        print(f'\nNow in iteration {iteration} (sample {sample_id}) ...\n')
        p_exts = np.NaN*np.ones(measure_intervals)
        ext_times_list = []
        
        for interval_id in range(measure_intervals):
            try:
                print(f'Now in time {interval_id} in ensemble interval ...')
                lattice = EvolvingLattice(L, *prob_params)
                p_ext, ext_times = lattice.measure_pext(
                    lattice=hists[sample_id][-start_from_end + interval_id].copy(), 
                    T=fin_gen, 
                    ensemble=ensemble_num
                )
                p_exts[interval_id] = p_ext
                ext_times_list.append(ext_times)
            except: pass
            
        np.savetxt(f'{save_dir}/sample{sample_id}_{init_or_last}_pext.txt', p_exts)
        
        with open(f'{save_dir}/sample{sample_id}_{init_or_last}_ext_times.pkl', 'wb') as f: 
            dill.dump(ext_times_list, f)


# generate_samples_hist( ids=ids1, save_dir=save_dir1, hists=init_hists, init_or_last='init')

# generate_samples_hist( ids=ids2, save_dir=save_dir2, hists=init_hists, init_or_last='init')
# generate_samples_hist( ids=ids2, save_dir=save_dir2, hists=last_hists, init_or_last='last')

generate_samples_hist( ids=ids3, save_dir=save_dir3, hists=init_hists, init_or_last='init')
generate_samples_hist( ids=ids3, save_dir=save_dir3, hists=last_hists, init_or_last='last')





# num_plots = len(ids_)
# col_num = 5
# row_num, r = divmod(num_plots,col_num)
# if r != 0: row_num+=1
# plot_position = list(range(1,num_plots+1))

# fig = plt.figure()
# fig.set_size_inches(col_num*4,(row_num)*3)
# fig.suptitle('p_ext for each samples', fontsize=15)
# plt.axis('off')
# plot_id = -1

# for plot_id in range(num_plots):
#     ax = fig.add_subplot(row_num,col_num,plot_position[plot_id])
#     ax.set_title(f'sample {sample_id}')

#     intervals = np.linspace(start_from_end-1,
#                             start_from_end-measure_intervals,
#                             measure_intervals).astype(int)

#     ax.plot(intervals, p_exts)
#     ax.set_ylabel('p_ext',fontsize=10)
#     ax.set_xlabel('Checkpoints from extinct',fontsize=10)

#     fig.savefig(f'{save_dir}/masterfig.png')

# plt.close()