#! /usr/bin/env python3

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import sys
import glob
import logging
import argparse

import numpy as np

import database.v1.tools as tools


###-###-###-###-###-###-###- LOGGER -###-###-###-###-###-###-###

logger  = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler   = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s")
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def get_number_of_pretriggers(npz_files,
                              du,
                              thresh1=61,
                              thresh2=36,
                              samples_from_trace_edge=0):
    
    logger.info(f'Finding number of pretriggers for DU {du}')
    logger.info(f'Trigger parameters set to: thresh1 = {thresh1}, thresh2 = {thresh2}, samples_from_trace_edge = {samples_from_trace_edge}')
    
    n_files        = len(npz_files)
    n_entries_trig = np.zeros(n_files,dtype=int)
    n_entries_tot  = np.zeros(n_files,dtype=int)
    du_seconds     = np.zeros(1000000,dtype=int)
    files          = np.zeros(1000000,dtype='<U160')
    
    du_sec_idx = 0
    for i, file in enumerate(npz_files[:]):
        logger.info(f'Processing file {i+1}/{len(npz_files)}')
        logger.debug(f'{file}')

        with np.load(file) as f:
            du_ids  = f['du_ids']

            try:
                mask_du = tools.get_masks_du(du_ids)[du]
            except:
                logger.warning(f'No data of DU {du} in this file')
                continue

            traces_du = f['traces'][mask_du]
            traces_du = tools.filter_traces(traces_du,freq_highpass=50,freqs_notch=[50.2,55.1,126],bw_notch=[1.,1.,25.]) #freqs_notch=[50.2,55.1,126],bw_notch=[1.,1.,25.]

            du_seconds_du = f['du_seconds'][mask_du]

            n_entries_tot[i] = len(traces_du)

            for trace, du_sec in zip(traces_du,du_seconds_du):
                _, trigger_flag, _ = tools.thresh_trigger(trace,threshold1=thresh1,threshold2=thresh2,samples_from_trace_edge=samples_from_trace_edge)

                if trigger_flag != 0:
                    n_entries_trig[i] += 1
                    du_seconds[du_sec_idx] = du_sec
                    files[du_sec_idx] = file
                    du_sec_idx += 1
                    

        logger.debug(f'n_entries_trig = {n_entries_trig[i]}, n_entries_tot = {n_entries_tot[i]}')
        logger.debug(f'{files[du_sec_idx]}')

    logger.info(f'Found {n_entries_trig.sum()} pretriggers in {n_entries_tot.sum()} traces')
    
    return n_entries_trig, n_entries_tot, du_seconds[:du_sec_idx], files[:du_sec_idx]


def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Selects GP13 data in GrandRoot format and saves the traces in npz format.")

    parser.add_argument('input_dir',
                        type=str,
                        help='Input directory (recursive) containing GP13 data in GrandRoot format.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='./',
                        help='Directory where the GP13 traces will be stored in npz format.')
    
    parser.add_argument('-du',
                        '--du_ref',
                        type=int,
                        default=1017,
                        help='Reference DU for which to test the pre-trigger thresholds.')
    
    parser.add_argument('-th1',
                        '--thresh1',
                        type=int,
                        default=61,
                        help='Threshold T1 for the pre-trigger.')
    
    parser.add_argument('-th2',
                        '--thresh2',
                        type=int,
                        default=36,
                        help='Threshold T2 for the pre-trigger.')
    
    parser.add_argument('-sfe',
                        '--samples_from_trace_edge',
                        type=int,
                        default=100,
                        help='Reference DU for which to test the pre-trigger thresholds.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    logger.info('*** START OF SCRIPT ***')

    #-#-#- Get parser arguments -#-#-#
    args                    = manage_args()
    input_dir               = args.input_dir
    output_dir              = args.output_dir
    du_ref                  = args.du_ref
    thresh1                 = args.thresh1
    thresh2                 = args.thresh2
    samples_from_trace_edge = args.samples_from_trace_edge

    #-#-#- Get npz files from input directory -#-#-#
    input_files = sorted( glob.glob( os.path.join(input_dir,'*.npz') ) )

    #-#-#- Get number of triggers for reference DU and save as npz file in output directory -#-#-#
    n_entries_trig, n_entries_tot, du_seconds, files = get_number_of_pretriggers(input_files,
                                                                                 du_ref,
                                                                                 thresh1=thresh1,
                                                                                 thresh2=thresh2,
                                                                                 samples_from_trace_edge=samples_from_trace_edge)

    output_filename = f'trigger_count_du_{du_ref}_th1_{thresh1}_th2_{thresh2}_sfe_{samples_from_trace_edge}.npz'
    output_file     = os.path.join(output_dir,output_filename)
    
    np.savez(output_file,
             n_entries_trig=n_entries_trig,
             n_entries_tot=n_entries_tot,
             du_seconds=du_seconds,
             input_dir=input_dir,
             files=files)
    
    logger.info(f'Saved pre-trigger results in {output_file}')
    
    logger.info('*** END OF SCRIPT ***')

