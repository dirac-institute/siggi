## Example LSST+1 script
# First import code
import sys
sys.path.append('../..')
import os
import time
from siggi import siggi, filters, spectra, Sed
from siggi.lsst_utils import BandpassDict, Bandpass, PhotometricParameters
import numpy as np

def prior_z(z):
    #normalization = 0.08333
    a, b, z0 = np.array([ 0.7787937 ,  2.47523432,  1.47403178])
    return (z**a)*np.exp(-(z/z0)**b)

if __name__ == "__main__":

    f = filters()
    s = spectra()

    sed_list = []
    for sed_name in os.listdir('../../data/cww_kin_lephare/'):
        sed_obj = Sed()
        sed_obj.readSED_flambda('../../data/cww_kin_lephare/%s' % sed_name)
        # Convert from angstroms to nm
        sed_obj.wavelen /= 10.
        sed_list.append(sed_obj)

    num_filters = 1
    total_non_zero = 0
    num_trials = 510

    bp_list = []
    new_phot_params = {}
    bp_dir = '/astro/store/epyc/users/brycek/siggi/siggi/data/lsst_baseline_throughputs'
    for filter_name in ['u', 'g', 'r', 'i', 'z', 'y']:
        current_bp = Bandpass()
        print(os.path.join(bp_dir, 'filter_%s.dat' % filter_name))
        current_bp.readThroughput(os.path.join(bp_dir, 'filter_%s.dat' % filter_name))
        bp_list.append(current_bp)
        new_phot_params[filter_name] = PhotometricParameters(bandpass=filter_name)

    new_phot_params['filter_0'] = PhotometricParameters(nexp=160*2, bandpass='filter_6')
    #bp_list.append(new_filt['filter_0'])
    frozen_dict = BandpassDict(bp_list, ['u', 'g', 'r', 'i', 'z', 'y'])#, 'filter_0'])

    sed_weights = np.ones(len(sed_list))/len(sed_list)

    sig_example = siggi(sed_list,
                        sed_weights, prior_z,
                        z_min=0.00, z_max=2.3, z_steps=47, phot_params=new_phot_params)

    x0 = None#[[300.,         414.99703239, 562.22744686, 630.98517889 ],
          #[ 318.40555564,  408.17312354,  870.49778848, 1073.39207548] ]
    y0 = None

    start = time.time()
    print('Starting at ', time.localtime())

    rand_state = np.random.RandomState(1305)
    ratio = None

    res = sig_example.optimize_filters(num_filters=num_filters,
                                       filt_min=300., filt_max=1100.,
                                       set_ratio=ratio,
                                       procs=2, n_opt_points=num_trials,
                                       system_wavelen_min=300.,
                                       system_wavelen_max=1200.,
                                       starting_points=x0,
                                       frozen_filt_dict=frozen_dict,
                                       frozen_filt_eff_wavelen=[365., 477., 622., 765., 870., 1015],
                                       #acq_func_kwargs_dict={'kappa':3.5},
                                       set_width=10,
                                       optimizer_verbosity=10,
                                       rand_state=rand_state,
                                       save_optimizer='frozen_filter_opt.pkl')

    trial_vals = np.array(res.yi)
    trial_pts = np.array(res.Xi)
    best_val = np.min(res.yi)
    best_pt = trial_pts[np.argmin(res.yi)]
    random_pts_used = res.random_pts_used

    non_zero_pts = np.where(trial_vals != 0.)[0]
    total_non_zero = len(non_zero_pts)

    suffix = 'ff_10sed_catsim_long'

    with open('results/run_results_%s.txt' % suffix, 'w') as f:
        f.write('After %i trials: \n' % total_non_zero)
        f.write('Random Points Used: %i \n' % random_pts_used)
        f.write('Best point: ')
        for pt_val in best_pt:
            f.write('%.4f ' % pt_val)
        f.write('\n')
        f.write('Best Information Gain: %.4f' % (-1.*best_val))

    np.savetxt('results/run_points_%s.txt' % suffix, trial_pts, fmt='%f')
    np.savetxt('results/run_values_%s.txt' % suffix, -1.*np.array(trial_vals), fmt='%f')


    finish = time.time()

    print('Job finished in %.4f seconds.' % (finish-start))

    print(best_pt, -1.*best_val)
