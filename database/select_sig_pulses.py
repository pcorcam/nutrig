#! /usr/bin/env python3
'''
Selects pulses in different ranges of zenith and opening angle w.r.t. the shower axis.
'''

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import glob
import logging
import argparse
import random

import numpy as np

import tools



logger = logging.getLogger(__name__)


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description='Randomly selects pulses in specified bins of zenith and opening angle w.r.t. shower axis.')

    parser.add_argument('input_dir',
                        type=str,
                        help='Input directory containing files with signal pulses in npz format.')
    
    parser.add_argument('-od',
                        '--output_dir',
                        type=str,
                        default=None,
                        help='Directory where the signal pulse files will be stored.')
    
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=300, # for GP300 :)
                        help='Seed for random number generator.')
    
    parser.add_argument('-nbz',
                        '--n_bins_zenith',
                        type=int,
                        default=10, # for GP300 :)
                        help='Number of bins for zenith.')
    
    parser.add_argument('-nbo',
                        '--n_bins_omega',
                        type=int,
                        default=10, # for GP300 :)
                        help='Number of bins for omega_diff = |omega-omega_c|/omega_c.')
    
    parser.add_argument('-ntb',
                        '--n_traces_per_bin',
                        type=int,
                        default=100, # for GP300 :)
                        help='Number of traces to target in each bin.')
    
    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')
    
    return parser.parse_args()

###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Get parser arguments -#-#-#
    args             = manage_args()
    input_dir        = args.input_dir
    output_dir       = args.output_dir
    seed             = args.seed
    n_bins_zenith    = args.n_bins_zenith
    n_bins_omega     = args.n_bins_omega
    n_traces_per_bin = args.n_traces_per_bin
    verbose          = args.verbose

    logger = tools.load_logger(verbose)
    logger.info('*** START OF SCRIPT ***')

    #-#-#- Set the histogram ranges -#-#-#

    # zenith and omega_diff ranges are fixed
    # see also .../nutrig/template_lib/make_template_lib.ipynb
    bin_edges_zenith = np.linspace(30.60,87.35,n_bins_zenith+1)
    bin_edges_omega  = np.linspace(0,2,n_bins_omega+1)

    #-#-#- Get input files and shuffle them with random number generator -#-#-#
    input_files = np.array( sorted( glob.glob( os.path.join( input_dir,'*.npz' ) ) ) )
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(input_files)

    #-#-#- Initiate arrays to fill -#-#-#
    n_traces_tot           = n_traces_per_bin*n_bins_zenith*n_bins_omega
    traces_output          = np.zeros( (n_traces_tot,3,1024),dtype=int )
    zenith_output          = np.zeros( (n_traces_tot,) )
    omega_diff_output      = np.zeros( (n_traces_tot,) )
    pretrig_flags_output   = np.zeros( (n_traces_tot,) )
    pretrig_times_output   = np.zeros( (n_traces_tot,) )
    inj_pulse_times_output = np.zeros( (n_traces_tot,) )
    snr_output             = np.zeros( (n_traces_tot,3) )
    entries_bin            = np.zeros( (n_bins_zenith,n_bins_omega) ) # entries in each zenith-omega_diff bin

    logger.info(f'####################################################################################')
    logger.info(f'Selecting {n_traces_tot} from simulations in {input_dir} with the following options:')
    logger.info(f'{n_bins_zenith} bins for zenith: {bin_edges_zenith}')
    logger.info(f'{n_bins_omega} bins for |omega-omega_c|/omega_c: {bin_edges_omega}')
    logger.info(f'Maximum {n_traces_per_bin} traces per bin')
    logger.info(f'####################################################################################')

    #-#-#- Loop over all input files -#-#-#
    k = 0
    for i, input_file in enumerate(input_files[:]):
        logger.info(f'Processing file {i+1}/{len(input_files)}: {input_file}')

        f               = np.load(input_file)
        zenith          = f['zenith']
        omega           = f['omega']
        omega_c         = f['omega_c']
        pretrig_flags   = f['pretrig_flags']
        pretrig_times   = f['pretrig_times']
        snr             = tools.get_snr(f['traces'],f['inj_pulse_times'].astype(int))
        snr             = np.max(snr[:,:2],axis=1) # Take the largest SNR of X and Y as the SNR parameter
        omega_diff      = np.abs(omega-omega_c)/omega_c # relative difference of antenna-core opening angle w.r.t. Cherenkov angle
        pretrig_idcs    = np.where(pretrig_flags>0)[0] # indices where there was an FLT-0 pretrigger
        #flag_bin        = np.zeros( (n_bins_zenith,n_bins_omega),dtype=bool ) # flag for entry in each zenith-omega_diff bin for this file

        zenith_bin_idx = np.where( np.logical_and( zenith >= bin_edges_zenith[:-1],zenith < bin_edges_zenith[1:] ) )[0] # find the corresponding zenith bin

        rng.shuffle(pretrig_idcs) # add extra randomness layer

        #-#-#- Loop over all entries with a pretrigger flag; each file can only contribute 1 trace per 1 bin (reduces effect of a few big showers dominating the sample) -#-#-#
        for pretrig_idx in pretrig_idcs:
            omega_bin_idx = np.where( omega_diff[pretrig_idx] >= bin_edges_omega )[0][-1] # find the corresponding omega_diff bin

            if snr[pretrig_idx] < 4 or snr[pretrig_idx] >= 6: # only keep events with an SNR of 5-8
                continue
            if np.any(omega_bin_idx >= n_bins_omega): # skip if relative omega difference is larger than highest bin (rare but can happen)
                continue
            if entries_bin[zenith_bin_idx,omega_bin_idx] == n_traces_per_bin: # skip if there are enough entries in the bin
                continue
            # if flag_bin[zenith_bin_idx,omega_bin_idx] == True: # skip if this file has already yielded an entry for this bin; ensures enough variety of simulations
            #     continue

            traces_output[k]          = f['traces'][pretrig_idx]
            zenith_output[k]          = zenith
            omega_diff_output[k]      = omega_diff[pretrig_idx]
            pretrig_flags_output[k]   = pretrig_flags[pretrig_idx]
            pretrig_times_output[k]   = pretrig_times[pretrig_idx]
            snr_output[k]             = snr[pretrig_idx]

            entries_bin[zenith_bin_idx,omega_bin_idx] += 1
            #flag_bin[zenith_bin_idx,omega_bin_idx]     = True
            k += 1

            break # If you have one entry, break this loop and go to the next file

        logger.info(f'Total number of traces collected: {k}/{n_traces_tot}')

        if k == n_traces_tot:
            break

    #-#-#- Save traces in output file -#-#-#
    if output_dir is None:
        output_dir = '/sps/grand/pcorrea/nutrig/database/sig/'

    output_filename = f'sig_dataset_bias_test_bins_{n_bins_zenith}x{n_bins_omega}_zenith_{bin_edges_zenith[0]}_{bin_edges_zenith[-1]}_omega_diff_{bin_edges_omega[0]}_{bin_edges_omega[-1]}_seed_{seed}.npz'
    output_file     = os.path.join(output_dir,output_filename)

    np.savez(output_file,
             traces=traces_output[:k],
             zenith=zenith_output[:k],
             omega_diff=omega_diff_output[:k],
             pretrig_flags=pretrig_flags_output[:k],
             pretrig_times=pretrig_times_output[:k],
             snr=snr_output[:k])

    logger.info(f'Number of entries reached in each zenith-omega_diff bin:\n {entries_bin}')
    logger.info(f'Saved {k} traces at: {output_file}')
    logger.info('*** END OF SCRIPT ***')