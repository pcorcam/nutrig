#! /usr/bin/env python3

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import glob
import logging
import argparse

import numpy as np

import database.tools as tools


logger = logging.getLogger(__name__)


'''

1 Calculate how many traces you need - should be the same as train+test sample of bkg pulses
2 Create bins of SNR: 3-8 in bins of .25? would be 20-> ~400 traces per bin
3 Loop over signal files
4 Take a random file
5 Take a random trace
6 Add trace in SNR bin if not full. SNR is defined as peak/RMS all trace (?)

'''


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def get_snr_with_file_tags(sig_files,
                           max_snr=10):

    logger.info(f'Loading SNR values from {len(sig_files)} signal files in {os.path.dirname(sig_files[0])}')
    logger.info(f'Only keeping SNR values < {max_snr}')

    snr        = np.zeros(1000000)
    file_idx   = np.zeros(1000000,dtype=int)
    file_entry = np.zeros(1000000,dtype=int)
    energy     = np.zeros(1000000)

    n_entries_tot = 0
    for i, sig_file in enumerate(sig_files[:]):
        logger.info(f'Loading SNRs in file {i+1}/{len(sig_files)}')

        with np.load(sig_file) as f:
            mask_trig = np.where(f['pretrig_flags']==0,False,True)
            snr_new   = np.max(f['snr'][...,:2],axis=1) # SNR of event = maximum of X/Y SNR (= where you triggered)
            mask_snr  = np.where(snr_new<=max_snr,True,False)
            
            mask  = np.logical_and(mask_snr,mask_trig)

            if np.any(mask):
                entries_mask   = np.arange( len(mask) )[mask]
                n_entries_mask = len( entries_mask )

                # snr        = np.hstack( ( snr, snr_new[mask] ) )
                # file_idx   = np.hstack( ( file_idx, i*np.ones(n_entries_mask) ) )
                # file_entry = np.hstack( ( file_entry, entries_mask ) )

                entries_new = np.arange(n_entries_tot,n_entries_tot+n_entries_mask)

                snr[entries_new]        = snr_new[mask]
                file_idx[entries_new]   = i*np.ones(n_entries_mask,dtype=int)
                file_entry[entries_new] = entries_mask.astype(int)

                energy[entries_new] = float(os.path.basename(sig_file).split('_')[7]) * np.ones(entries_new.shape)

                n_entries_tot += n_entries_mask

    logger.info(f'Loaded {n_entries_tot} SNR entries')

    return snr[:n_entries_tot], file_idx[:n_entries_tot], file_entry[:n_entries_tot], energy[:n_entries_tot]


def select_trace_idcs(snr,
                      n_select,
                      bin_edges=np.arange(3.,8.26,.25),
                      energy=None,
                      rng=np.random.default_rng()):
    
    logger.info(f'Selecting {n_select} random trace indices based on their SNR')
    logger.info(f'Binning the SNR in: {bin_edges[:-1]}')

    #-#-#- Find which bin each SNR belongs to -#-#-#
    which_bin = np.digitize(snr,bins=bin_edges)

    #-#-#- Sort the SNR entries with increasing bin number -#-#-#
    snr_idcs_sorted  = np.argsort(which_bin)
    which_bin_sorted = which_bin[snr_idcs_sorted]

    #-#-#- Get rid of entries that are not within the bins of the histogram -#-#-#
    mask_hist      = np.logical_and(which_bin_sorted!=0,which_bin_sorted!=len(bin_edges))
    snr_idcs_hist  = snr_idcs_sorted[mask_hist]
    which_bin_hist = which_bin_sorted[mask_hist]

    #-#-#- Find how many entries there are per bin and which SNR index correspond to each edge -#-#-#
    bin_numbers, bin_idcs, bin_counts = np.unique(which_bin_hist,return_index=True,return_counts=True)

    #-#-#- Compute how many entries you would ideally like per bin -#-#-#
    if type(energy) is type(None):
        power_law_index = -0.09
        logger.warning(f'No energy weights specified! Weighting the SNR bins with the "empirical" function y = 10^({power_law_index}x)')
        bin_weights  = 10**(power_law_index*bin_edges[:-1]) # see `signal_selection.ipynb`
    else:
        weights_energy  = ( energy/np.min(energy) )**-3
        bin_weights,_ = np.histogram(snr,bins=bin_edges,weights=weights_energy,density=True)
        logger.info(f'Weighting the SNR entries with an E^-3 spectrum')

    bin_weights /= bin_weights.sum()
    logger.debug(f'Normalized bin weights: {bin_weights}')

    n_per_bin = n_select*bin_weights
    n_per_bin = n_per_bin.astype(int)

    diff = int( n_select - n_per_bin.sum() )
    n_per_bin[:diff] += 1

    logger.debug(f'TARGETED number of entries per bin: {n_per_bin}')

    #-#-#- Select random SNR indices per bin -#-#-#
    snr_idcs_rand = np.zeros(0,dtype=int)
    for i in range(len(bin_numbers)):
        bin_start           = bin_idcs[i]
        bin_end             = bin_idcs[i]+bin_counts[i]
        snr_idcs_for_choice = snr_idcs_hist[bin_start:bin_end] #np.arange(snr_idx_start,snr_idx_end)

        #-#-#- If there are not enough entries to select from, get all of them and compensate with higher-SNR bins -#-#-#
        # NOTE: implicitly assumes that bins with higher SNRs have enough entries to compensate
        if len(snr_idcs_for_choice) < n_per_bin[i]:
            logger.warning(f'Not enough statistics in bin number {i}. Will compensate with higher-SNR bins.')

            n_missing            = n_per_bin[i] - len(snr_idcs_for_choice)
            n_tot_compensate     = np.sum(n_per_bin[i+1:]) + n_missing
            n_compensate_per_bin = n_missing * (bin_weights[i+1:]/bin_weights[i+1:].sum())
            n_per_bin[i+1:]     += n_compensate_per_bin.astype(int)

            diff = int( n_tot_compensate - np.sum(n_per_bin[i+1:])  )
            n_per_bin[i+1:i+1+diff] += 1
        
            n_per_bin[i]  = len(snr_idcs_for_choice) # for bookkeeping only
            snr_idcs_rand = np.hstack( (snr_idcs_rand,snr_idcs_for_choice) )

        #-#-#- If there are enough indices to select from, select them at random -#-#-#
        else:
            snr_idcs_rand_new = rng.choice( snr_idcs_for_choice,size=n_per_bin[i],replace=False)
            snr_idcs_rand     = np.hstack( (snr_idcs_rand,snr_idcs_rand_new) )

    logger.debug(f'SELECTED number of entries per bin: {n_per_bin}')
    
    if n_per_bin.sum() < n_select:
        logger.warning(f'Not enough statistics. Only selected {n_per_bin.sum()} number of entries.')

    return snr_idcs_rand


def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description='Makes two data files of signal pulses (training+testing) for trigger studies.')

    parser.add_argument('input_dir',
                        type=str,
                        help='Input directory containing files with signal pulses in npz format.')

    parser.add_argument('bkg_file_train',
                        type=str,
                        help='Path to the the background-pulse file used for training made using `make_bkg_pulse_file.py`.')
    
    parser.add_argument('-od',
                        '--output_dir',
                        type=str,
                        default=None,
                        help='Directory where the signal pulse files will be stored.')
    
    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Get parser arguments -#-#-#
    args            = manage_args()
    input_dir       = args.input_dir
    bkg_file_train  = args.bkg_file_train
    output_dir      = args.output_dir
    verbose         = args.verbose

    logger = tools.load_logger(verbose)
    logger.info('*** START OF SCRIPT ***')

    #-#-#- Set I/O parameters -#-#-#
    if output_dir == None:
        logger.warning('Setting output_dir to input_dir')
        output_dir = input_dir

    input_files = sorted( glob.glob( os.path.join(input_dir,'*.npz') ) )

    #-#-#- Get number of traces from background-pulse files -#-#-#
    with np.load(bkg_file_train) as f:
        n_traces_train = len(f['traces'])
        logger.info(f'Will select {n_traces_train} signal traces for training sample')

    bkg_file_test = bkg_file_train.replace('train','test')
    with np.load(bkg_file_test) as f:
        n_traces_test = len(f['traces'])
        logger.info(f'Will select {n_traces_test} signal traces for testing sample')

    #-#-#- Get the signal filenames and the RNG seed from the background filenames -#-#-#
    sig_filename_train = os.path.basename(bkg_file_train).replace('bkg','sig')
    sig_filename_test  = os.path.basename(bkg_file_test).replace('bkg','sig')

    split = os.path.splitext(sig_filename_train)[0].split('_')
    seed  = int(split[-1])
    rng   = np.random.default_rng(seed=seed)
    logger.info(f'Set the seed for the RNG to: {seed}')

    #-#-#- Load the SNRs and file tags from the input files -#-#-#
    snr, file_idx, file_entry, energy = get_snr_with_file_tags(input_files)

    #-#-#- Select random trace indices based on the SNR  -#-#-#
    idcs_sel = select_trace_idcs(snr,n_traces_train+n_traces_test,rng=rng,energy=energy)
    snr_sel  = snr[idcs_sel]

    #-#-#- Select the traces from the input files  -#-#-#
    sig_traces_sel  = np.zeros( (len(idcs_sel),3,1024),dtype=int)
    inj_pulse_times = np.zeros( (len(idcs_sel), ),dtype=int)
    pretrig_flags   = np.zeros( (len(idcs_sel), ),dtype=int)
    pretrig_times   = np.zeros( (len(idcs_sel), ),dtype=int)
    files           = np.zeros( (len(idcs_sel), ),dtype='<U200')
    entries         = np.zeros( (len(idcs_sel), ),dtype=int)

    logger.info(f'Fetching information from {len(idcs_sel)} selected traces')
    for k, idx_sel in enumerate(idcs_sel):
        input_file = input_files[file_idx[idx_sel]]
        files[k]   = input_file

        entry      = file_entry[idx_sel]
        entries[k] = entry

        with np.load(input_file) as f:
            sig_traces_sel[k]  = f['traces'][entry]
            inj_pulse_times[k] = f['inj_pulse_times'][entry]
            pretrig_times[k]   = f['pretrig_times'][entry]
            pretrig_flags[k]   = f['pretrig_flags'][entry]

    #-#-#- Devide the traces randomly into a training sample and a test sample -#-#-#
    output_file_train = os.path.join(output_dir,sig_filename_train)
    output_file_test  = os.path.join(output_dir,sig_filename_test)

    n_traces  = len(sig_traces_sel)
    mask_rand = np.arange(n_traces)
    idx_split = int(n_traces/2)

    np.random.seed(seed=seed)
    np.random.shuffle(mask_rand)

    #-#-#- Save the training and test samples -#-#-#
    logger.info(f'Saving {len(mask_rand[:idx_split])} signal pulses for TRAINING in {output_file_train}')

    np.savez(output_file_train,
             traces=sig_traces_sel[mask_rand[:idx_split]],
             snr=snr_sel[mask_rand[:idx_split]],
             inj_pulse_times=inj_pulse_times[mask_rand[:idx_split]],
             pretrig_flags=pretrig_flags[mask_rand[:idx_split]],
             pretrig_times=pretrig_times[mask_rand[:idx_split]],
             entries=entries[mask_rand[:idx_split]],
             files=files[mask_rand[:idx_split]])

    logger.info(f'Saving {len(mask_rand[idx_split:])} signal pulses for TESTING in {output_file_test}')

    np.savez(output_file_test,
             traces=sig_traces_sel[mask_rand[idx_split:]],
             snr=snr_sel[mask_rand[idx_split:]],
             inj_pulse_times=inj_pulse_times[mask_rand[idx_split:]],
             pretrig_flags=pretrig_flags[mask_rand[idx_split:]],
             pretrig_times=pretrig_times[mask_rand[idx_split:]],
             entries=entries[mask_rand[idx_split:]],
             files=files[mask_rand[idx_split:]])
    
    logger.info('*** END OF SCRIPT ***')