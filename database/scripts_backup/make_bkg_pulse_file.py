#! /usr/bin/env python3

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import glob
import logging
import argparse

import numpy as np

import tools


logger = logging.getLogger(__name__)


'''

'''


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description='Makes two data files of background pulses (training+testing) for trigger studies.')

    parser.add_argument('input_dir',
                        type=str,
                        help='Input directory containing files with background pulses in npz format.')

    parser.add_argument('output_filename',
                        type=str,
                        help='Basename of output files to store the background pulses.')
    
    parser.add_argument('-od',
                        '--output_dir',
                        type=str,
                        default=None,
                        help='Directory where the background pulse file will be stored.')
    
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=300, # for GP300 :)
                        help='Seed for random number generator.')
    
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
    output_filename = args.output_filename 
    output_dir      = args.output_dir
    seed            = args.seed
    verbose         = args.verbose

    logger = tools.load_logger(verbose)
    logger.info('*** START OF SCRIPT ***')
    
    #-#-#- Set I/O parameters -#-#-#
    if output_dir == None:
        logger.warning('Setting output_dir to input_dir')
        output_dir = input_dir

    input_files = glob.glob( os.path.join( input_dir,'*.npz' ) )

    #-#-#- Merge traces from input files -#-#-#
    logger.info(f'Collecting background pulses from {input_dir}...')

    traces        = np.zeros( (0,3,1024),dtype=int ) # assume trace length is 1024
    pretrig_times = np.zeros( (0,),dtype=int )
    entries       = np.zeros( (0,),dtype=int )
    files         = np.zeros( (0,),dtype='<U100' )

    # Slow but does the job; ok for < 1e4 pulses
    for i, input_file in enumerate(input_files):
        logger.info(f'Processing input file {i+1}/{len(input_files)}...')
        
        f             = np.load(input_file)

        traces        = np.vstack( (traces,f['traces']) )
        pretrig_flags = np.hstack( (pretrig_flags,f['pretrig_flags']) )
        pretrig_times = np.hstack( (pretrig_times,f['pretrig_times']) )
        entries       = np.hstack( (entries,f['entries']) )

        file_arr      = np.array( [ input_file for i in range( len(f['entries'] ) ) ],dtype='<U100' )
        files         = np.hstack( (files,file_arr) )

    #-#-#- Devide the traces randomly into a training sample and a test sample -#-#-#
    output_file_train = os.path.join(output_dir,output_filename+f'_train_seed_{seed}.npz')
    output_file_test  = os.path.join(output_dir,output_filename+f'_test_seed_{seed}.npz')

    n_traces  = len(traces)
    mask_rand = np.arange(n_traces)
    idx_split = int(n_traces/2)

    np.random.seed(seed=seed)
    np.random.shuffle(mask_rand)

    #-#-#- Save the training and test samples -#-#-#
    logger.info(f'Saving {len(mask_rand[:idx_split])} background pulses for TRAINING in {output_file_train}')

    np.savez(output_file_train,
             traces=traces[mask_rand[:idx_split]],
             pretrig_flags=pretrig_flags[mask_rand[:idx_split]],
             pretrig_times=pretrig_times[mask_rand[:idx_split]],
             entries=entries[mask_rand[:idx_split]],
             files=files[mask_rand[:idx_split]])

    logger.info(f'Saving {len(mask_rand[idx_split:])} background pulses for TESTING in {output_file_test}')

    np.savez(output_file_test,
             traces=traces[mask_rand[idx_split:]],
             pretrig_flags=pretrig_flags[mask_rand[idx_split:]],
             pretrig_times=pretrig_times[mask_rand[idx_split:]],
             entries=entries[mask_rand[idx_split:]],
             files=files[mask_rand[idx_split:]])
    
    logger.info('*** END OF SCRIPT ***')