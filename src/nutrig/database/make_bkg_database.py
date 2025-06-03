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

'''

###-###-###-###-###-###-###- FIXED PARAMETERS -###-###-###-###-###-###-###

'''
Filters and thresholds are fixed based on analysis with
`test_pretrigger.py` and `find_pretrigger_threshold.ipynb`
'''

freq_highpass = 50 # [MHz]
freqs_notch   = [50.2,55.1,126] # [MHz]
bw_notch      = [1.,1.,25.] # [MHz]

threshold1              = 35#55 # [ADC counts]
threshold2              = 25#35 # [ADC counts]
samples_from_trace_edge = 100 # [ADC samples]
include_Z               = False

du_ids_exclude = [1032,1085]


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description='Makes a database of background traces for offline trigger studies.')

    parser.add_argument('input_dir',
                        type=str,
                        help='Input directory of GP13 data in npz format.')
    
    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Get parser arguments -#-#-#
    args       = manage_args()
    input_dir  = args.input_dir
    verbose    = args.verbose

    logger = tools.load_logger(verbose)
    logger.info('*** START OF SCRIPT ***')

    #-#-#- Print out fixed parameters -#-#-# 
    logger.info('#########################################')
    
    logger.info('FILTER PARAMETERS')
    logger.info(f'freq_highpass = {freq_highpass} MHz')
    logger.info(f'freqs_notch   = {freqs_notch} MHz')
    logger.info(f'bw_notch      = {bw_notch} MHz')

    logger.info('THRESHOLD TRIGGER PARAMETERS')
    logger.info(f'threshold1 = {threshold1} ADC counts')
    logger.info(f'threshold2 = {threshold2} ADC counts')
    logger.info(f'samples_from_trace_edge = {samples_from_trace_edge} ADC samples')

    logger.info(F'EXCLUDING DUs {du_ids_exclude}')

    if not include_Z:
        logger.info('ONLY TRIGGERING ON X OR Y (NOT Z)')
    
    #-#-#- Create directories to save triggered and non-triggered traces -#-#-#
    trig_subdir = input_dir.replace('raw',f'pretrigger_pulses_th1_{threshold1}_th2_{threshold2}')
    stat_subdir = input_dir.replace('raw',f'pretrigger_stationary_th1_{threshold1}_th2_{threshold2}')

    logger.info(f'Traces that pass the threshold trigger will be saved in {trig_subdir}')
    logger.info(f'Traces that do not pass the threshold trigger will be saved in {stat_subdir}')
    logger.info('#########################################')

    #-#-#- Loop over all npz files in dir -#-#-#
    input_files = sorted( glob.glob( os.path.join(input_dir,'*.npz') ) )

    for i, input_file in enumerate(input_files[:]):
        f       = np.load(input_file)
        traces  = f['traces']
        du_ids  = f['du_ids']

        n_traces = traces.shape[0]
        entries  = np.arange(n_traces)

        logger.info(f'Processing {n_traces} traces in input file {i+1}/{len(input_files)}...')

        #-#-#- Get the masks for DU IDs to exlude -#-#-#
        masks_du        = tools.get_masks_du(du_ids)
        mask_du_exclude = np.ones(n_traces,dtype=bool)

        for du_id in du_ids_exclude:
            if np.any(du_ids) == du_id:
                mask_du_exclude = np.logical_and( mask_du_exclude,np.logical_not( masks_du[du_id] ) )

        #-#-#- Perform the threshold trigger on all FILTERED traces -#-#-#
        traces_filtered = tools.filter_traces(traces,
                                              freq_highpass=freq_highpass,
                                              freqs_notch=freqs_notch,
                                              bw_notch=bw_notch)
        
        pretrig_flags, pretrig_times = tools.find_thresh_triggers(traces_filtered,
                                                                  threshold1=threshold1,
                                                                  threshold2=threshold2,
                                                                  samples_from_trace_edge=samples_from_trace_edge,
                                                                  include_Z=include_Z)
        
        #-#-#- Set masks for triggered and "stationary" traces taking into account DUs to exclude -#-#-#
        mask_flag = np.where(pretrig_flags==0,False,True)
        mask_trig = np.logical_and( mask_flag,mask_du_exclude )
        mask_stat = np.logical_and( np.logical_not( mask_trig ),mask_du_exclude )

        #-#-#- Save triggered traces (FILTERED) with corresponding trigger time -#-#-#
        if np.any(mask_trig):
            logger.info(f'Found {len(pretrig_times[mask_trig])} triggers!')

            out_file_trig = os.path.join( trig_subdir,os.path.basename(input_file).replace('.npz','_trig_filt.npz') )

            np.savez(out_file_trig,
                     traces=traces_filtered[mask_trig],
                     pretrig_flags=pretrig_flags[mask_trig],
                     pretrig_times=pretrig_times[mask_trig],
                     du_ids=du_ids[mask_trig],
                     entries=entries[mask_trig])

            logger.debug(f'Saved triggered traces in {out_file_trig}')

        #-#-#- Save "stationary" traces (NOT FILTERED) -#-#-#
        mask_stat     = np.logical_not(mask_trig)
        out_file_stat = os.path.join( stat_subdir,os.path.basename(input_file).replace('.npz','_stat.npz') )

        np.savez(out_file_stat,
                 traces=traces[mask_stat],
                 du_ids=du_ids[mask_stat],
                 entries=entries[mask_stat])

        logger.debug(f'Saved stationary traces in {out_file_stat}')

    logger.info('*** END OF SCRIPT ***')