'''
This scripts creates a final database of background pulses for NUTRIG studies.
These will be stored in a single file.

The following steps are followed:
    1. Take pulses from MD run 145 and CD runs 10083, 10085, 10086 that were selected with `make_bkg_data_base_v2_from_MD.py` and `make_bkg_data_base_v2_from_CD.py`.
    2. Make the selection based on SNR. Can be either uniform or "realistic" (TODO).
    3. Store the pulses and FLT-0 information.
'''

###########
# IMPORTS #
###########

import os
import glob
import logging
import argparse

import numpy as np

import grand.dataio.root_trees as rt

import database.tools as tools

logger = logging.getLogger(__name__)


####################
# FIXED PARAMETERS #
####################

RUNS     = [145,10083,10085,10086]
TH1      = 45
TH2      = 35
TQUIET   = 500
TPER     = 1000
TSEPMAX  = 200
NCMIN    = 2#1
NCMAX    = 10#7


CHANNELS_FLT0 = [0,1] # (0,1,2) = (X,Y,Z) ASSUMING FLOAT CHANNEL IS DISCARDED
MODE_FLT0     = 'OR' # Mode for the FLT-0. Can be 'OR' or 'AND'

CHANNEL_POL       = {0:'X',1:'Y',2:'Z'}
CHANNELS_FLT0_STR = ''
for ch in CHANNELS_FLT0:
    CHANNELS_FLT0_STR += CHANNEL_POL[ch]

N_SAMPLES  = 1024
N_CHANNELS = 3

SNR_BINS               = np.linspace(3,8,6)
TARGET_ENTRIES_PER_BIN = 1000
MODE_SNR               = 'UNIFORM'

BKG_DATABASE_BASEDIR = '/sps/grand/pcorrea/nutrig/database/v2/bkg'

OUT_DIR      = os.path.join(BKG_DATABASE_BASEDIR,'lib')
OUT_FILENAME = f'bkg_database_nutrig_v2_{MODE_SNR}.npz'
OUT_FILE     = os.path.join(OUT_DIR,OUT_FILENAME)

# For each event, we only want the data of one of the triggered DUs. Set the random generator's seed here
np.random.seed(80) # for GP80!


#############
# FUNCTIONS #
#############

def load_data(bkg_database_basedir,
              runs,
              channels_flt0=[0,1],
              mode_flt0='OR'):
    '''
    Load background pulses selected from GP80 data.

    Arguments
    ---------
    - `data_file`
        + type        : `str`
        + description : Absolute path to GP80 pulse directory.

    - `runs`
        + type        : `list[str]`
        + description : Run numbers to use for the final database construction.

    Returns
    -------

    '''

    data = {}

    data['traces']     = np.zeros( (0,N_CHANNELS,N_SAMPLES),dtype=int )
    data['FLT0_flags'] = np.zeros( (0,N_CHANNELS), dtype=bool )
    data['snr']        = np.zeros( (0,),dtype=float )
    data['t_pulse']    = np.zeros( (0,N_CHANNELS),dtype=int )


    for run in runs:
        logger.info(f'Loading background pulses from run {run}...')

        bkg_database_rundir           = os.path.join(bkg_database_basedir,f'GP80_RUN_{run}_CH_{CHANNELS_FLT0_STR}_MODE_{MODE_FLT0}_TH1_{TH1}_TH2_{TH2}_TQUIET_{TQUIET}_TPER_{TPER}_TSEPMAX_{TSEPMAX}_NCMIN_{NCMIN}_NCMAX_{NCMAX}')
        path_metadata_file            = os.path.join(bkg_database_rundir,'metadata.npz')
        bkg_database_rundir_filtered  = os.path.join(bkg_database_rundir,'filtered')
        paths_data_files_bkg_filtered = sorted( glob.glob( os.path.join(bkg_database_rundir_filtered,'*.npz') ) )

        with np.load(path_metadata_file,allow_pickle=True) as metadata_file:
            channels_flt0_run     = metadata_file['channels_flt0']
            mode_flt0_run         = metadata_file['mode_flt0']

        if np.all( channels_flt0_run != channels_flt0 ):
            logger.warning(f'Pulses from {bkg_database_rundir} were not triggered on the required FLT-0 channels, skipping...')
            continue

        if mode_flt0_run != mode_flt0:
            logger.warning(f'Pulses from {bkg_database_rundir} were not triggered in the required FLT-0 mode, skipping...')
            continue

        traces     = np.zeros( (0,N_CHANNELS,N_SAMPLES),dtype=int )
        FLT0_flags = np.zeros( (0,N_CHANNELS),dtype=bool )
        snr        = np.zeros( (0,),dtype=float)
        t_pulse    = np.zeros( (0,N_CHANNELS),dtype=int )

        for i, path_data_file_filtered in enumerate(paths_data_files_bkg_filtered):
            logger.info(f'Loading file {i+1}/{len(paths_data_files_bkg_filtered)}...')

            with np.load(path_data_file_filtered) as data_file_filtered:
                traces     = np.vstack( ( traces,data_file_filtered['traces'] ) )
                FLT0_flags = np.vstack( ( FLT0_flags,data_file_filtered['FLT0_flags'] ) )
                snr        = np.hstack( ( snr,data_file_filtered['snr'] ) )
                t_pulse    = np.vstack( ( t_pulse,data_file_filtered['t_pulse'] ) )

        data['traces']     = np.vstack( (data['traces'],traces.astype(int)) )
        data['FLT0_flags'] = np.vstack( (data['FLT0_flags'],FLT0_flags) )
        data['snr']        = np.hstack( (data['snr'],snr) )
        data['t_pulse']    = np.vstack( (data['t_pulse'],t_pulse.astype(int)) )
        
    return data


###########################
# DEFINE PARSER ARGUMENTS #
###########################

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Find background pulses in a GP80 MD data file.")

    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')
    
    return parser.parse_args()


################
# MAIN PROGRAM #
################

if __name__ == '__main__':
    parse_args = manage_args()

    logger = tools.load_logger(parse_args.verbose)
    logger.info('*** START OF SCRIPT ***')

    logger.info(f'*** Constructing NUTRIG background-pulse database file ***')
    logger.info(f'FLT-0 trigger settings of background-pulse pre selection:')
    logger.info(f'Channels: {np.array(["X","Y","Z"])[CHANNELS_FLT0]}')
    logger.info(f'Mode: {MODE_FLT0}')
    logger.info(f'TH1 = {TH1}, TH2 = {TH2}, TQUIET = {TQUIET}, TPER = {TPER}, TSEPMAX = {TSEPMAX}, NCMIN = {NCMIN}, NCMAX = {NCMAX}')
    logger.info(f'SNR mode for final selection: {MODE_SNR} in SNR')

    data          = load_data(BKG_DATABASE_BASEDIR,RUNS,channels_flt0=CHANNELS_FLT0,mode_flt0=MODE_FLT0)
    selected_data = tools.select_pulses_per_snr_bin(data,
                                                    snr_bins=SNR_BINS,
                                                    target_entries_per_bin=TARGET_ENTRIES_PER_BIN,
                                                    mode_snr=MODE_SNR,
                                                    n_channels=N_CHANNELS,
                                                    n_samples=N_SAMPLES)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    logger.info(f'Saving final background database at: {OUT_FILE}')

    np.savez(OUT_FILE,
             traces=selected_data['traces'],
             FLT0_flags=selected_data['FLT0_flags'],
             snr=selected_data['snr'],
             t_pulse=selected_data['t_pulse'])

    logger.info('*** END OF SCRIPT ***')