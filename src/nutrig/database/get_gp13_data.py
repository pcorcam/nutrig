#! /usr/bin/env python3

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import logging
import argparse

import numpy as np

import grand.dataio.root_trees as rt # type: ignore
from grand import manage_log # type: ignore


logger = logging.getLogger(__name__)


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###


def load_root_files(data_dir):
    '''
    '''
    logger.info(f'Recursively selecting all GrandRoot files in {data_dir}...')

    root_files = []
    filenames  = set()

    for root, dirs, files in os.walk(data_dir,topdown=False): # make sure topdown = False
        for name in files:
            if name in filenames: # make sure there are no duplicates
                continue
            root_files.append(os.path.join(root,name))
            filenames.add(name)

    logger.info(f'Selected {len(root_files)} GrandRoot files')

    return sorted(root_files)


def remove_root_files(root_files,
                      strings_rm):
    
    logger.info(f'Removing files with the following strings in the filename: {strings_rm}')
    
    root_files = np.array(root_files,dtype=str)
    mask       = np.ones(len(root_files),dtype=bool)

    for i, root_file in enumerate(root_files):
        for string in strings_rm:
            if string in os.path.basename(root_file):
                mask[i] = False
                break

    root_files_sel = root_files[mask]

    logger.info(f'Removed {len(mask[mask==False])} files')

    return root_files_sel.tolist() # make sure it's a list, otherwise GrandLib might bug


def convert_root_to_npz(root_files,
                        npz_dir,
                        float_ch=0):
    
    mask_ch = np.arange(4)
    if float_ch is not None:
        mask_ch = np.delete(mask_ch,float_ch)
    else:
        mask_ch = mask_ch[:-1]

    logger.info(f'Converting {len(root_files)} GrandRoot files to npz files')

    for i, root_file in enumerate(root_files[:]):
        if '4096points' in os.path.basename(root_file):
            n_samples = 4096
        else:
            n_samples = 1024

        logger.info(f'Converting GrandRoot file {i+1}/{len(root_files)}...')
        logger.debug(f'Float channel is set to {float_ch} and the number of trace samples is set to {n_samples}')
        
        with rt.DataFile(root_file) as df:
            tadc = df.tadc

            n_entries  = tadc.get_number_of_entries()
            traces     = np.zeros((n_entries,3,n_samples),dtype=int)
            du_ids     = np.zeros(n_entries,dtype=int)
            du_seconds = np.zeros(n_entries,dtype=int)

            for entry in range(n_entries):
                tadc.get_entry(entry)

                trace             = np.array(tadc.trace_ch[0])
                traces[entry]     = trace[mask_ch]
                du_ids[entry]     = tadc.du_id[0]
                du_seconds[entry] = tadc.du_seconds[0]

            #df.close()

        npz_filename = os.path.basename(root_file).replace('.root','.npz')
        npz_file     = os.path.join(npz_dir,npz_filename)

        np.savez(npz_file,
                 traces=traces,
                 du_ids=du_ids,
                 du_seconds=du_seconds)
        
        logger.info(f'File converted to npz!')
        logger.debug(f'File saved at {npz_file}')
    
    return


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

    parser.add_argument('--include_4096',
                        action='store_true',
                        default=False,
                        help='Include files with traces of 4096 samples.')
    
    parser.add_argument('-s',
                        '--start_date',
                        type=int,
                        default=20240116,
                        help='Start date (included) for the file selection, in YYYYMMDD format.')
    
    parser.add_argument('-e',
                        '--end_date',
                        type=int,
                        default=21000101,
                        help='End date (included) for the file selection, in YYYYMMDD format.')
    
    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    logger = manage_log.get_logger_for_script(__file__)

    #-#-#- Get parser arguments -#-#-#
    args         = manage_args()
    input_dir    = args.input_dir
    output_dir   = args.output_dir
    include_4096 = args.include_4096
    start_date   = args.start_date
    end_date     = args.end_date
    verbose      = args.verbose

    manage_log.create_output_for_logger(verbose,log_stdout=True)
    logger.info( manage_log.string_begin_script() )

    #-#-#- Load GrandRoot files in input directory -#-#-#
    root_files = load_root_files(input_dir) # sorted by date

    #-#-#- Remove unwanted files from selection -#-#-#
    strings_rm = []

    first_filename   = os.path.basename(root_files[0])
    first_date_files = int( first_filename.split('_')[1] )
    last_filename    = os.path.basename(root_files[-1])
    last_date_files  = int( last_filename.split('_')[1] )

    strings_rm += [str(date) for date in range(first_date_files,start_date)]
    strings_rm += [str(date) for date in range(end_date+1,last_date_files+1)] # +1 to include the end date and to exclude the last date in files

    if not include_4096:
        strings_rm.append('4096points')

    root_files_sel = remove_root_files(root_files,strings_rm)

    #-#-#- Convert GrandRoot files to npz files and save these in output directory -#-#-#
    convert_root_to_npz(root_files_sel,output_dir)

    logger.info( manage_log.string_end_script() )