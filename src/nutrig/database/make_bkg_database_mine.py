'''
This scripts creates a database of background pulses from the mine near the GP300 site.

The file we use to get mine events: `GP80_20250309_235256_RUN10070_CD_20dB_23DUs_GP43-ChY-X2X-Y2Y-CD-10000-22.root`
The 

The database format is equivalent to a "final" NUTRIG FLT-1 database.
That way it can be plugged in directly to the template FLT.
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

RUN_NUMBER          = 10070
DATA_DIR            = '/sps/grand/data/gp80/GrandRoot/2025/03'
DATA_FILE           = os.path.join( DATA_DIR,f'GP80_20250309_235256_RUN{RUN_NUMBER}_CD_20dB_23DUs_GP43-ChY-X2X-Y2Y-CD-10000-22.root' )
FLOAT_CH            = 0
N_CHANNELS          = 4
N_SAMPLES           = 1024
DU_IDS_EXCLUDE      = []
try:
    BASELINE_CALIB = tools.get_baseline_calib(baseline_dir=f'/sps/grand/pcorrea/gp80/baseline_calibration/RUN{RUN_NUMBER}')
except:
    BASELINE_CALIB = None

MINE_EVTS_DIR  = '/sps/grand/pcorrea/gp80/mine_analysis/'
MINE_EVTS_FILE = os.path.join( MINE_EVTS_DIR,'MINE_IDCS_' + os.path.basename(DATA_FILE).replace('.root','.txt') )

DICT_TRIGGER_PARAMS = {'th1'      : 80,#45, # [ADC counts]
                       'th2'      : 50,#35, # [ADC counts]
                       't_quiet'  : 500, # [ns]
                       't_period' : 1000, # [ns]
                       't_sepmax' : 200, # [ns]
                       'nc_min'   : 2,
                       'nc_max'   : 15}
SAMPLES_FROM_EDGE   = 100 # [ADC samples]
CHANNELS_FLT0       = [0,1] # (0,1,2) = (X,Y,Z) ASSUMING FLOAT CHANNEL IS DISCARDED
MODE_FLT0           = 'OR' # Mode for the FLT-0. Can be 'OR' or 'AND'

CHANNEL_POL       = {0:'X',1:'Y',2:'Z'}
CHANNELS_FLT0_STR = ''
for ch in CHANNELS_FLT0:
    CHANNELS_FLT0_STR += CHANNEL_POL[ch]


MODE_SNR             = 'MINE'

BKG_DATABASE_BASEDIR = '/sps/grand/pcorrea/nutrig/database/v2/bkg'

OUT_DIR      = os.path.join(BKG_DATABASE_BASEDIR,'lib')
OUT_FILENAME = f'bkg_database_nutrig_v2_{MODE_SNR}.npz'
OUT_FILE     = os.path.join(OUT_DIR,OUT_FILENAME)


#############
# FUNCTIONS #
#############

def load_data(data_file,
              mine_evts_file):
    '''
    Load selected data from a GP80 MD file.

    Arguments
    ---------
    - `data_file`
        + type        : `str`
        + description : Absolute path to GP80 MD file in GrandRoot format.

    - `mine_evts_file`
        + type        : `str`
        + description : Absolute path to data file with events identified as coming from the mine for the `data_file`.

    Returns
    -------
    - `traces`
        + type        : `np.ndarray[int]`
        + units       : ADC counts
        + description : Traces stored in the data file excluding the floating channel. Shape: `(N_entries,3,N_samples)`.

    - `du_ids`
        + type        : `np.ndarray[int]`
        + description : DU IDs corresponding to each trace entry.
    '''

    logger.info(f'Loading traces in {os.path.basename(data_file)}...')
    logger.info(f'For events from the mine as identified in: {mine_evts_file}')

    mine_event_idcs = np.loadtxt(mine_evts_file, delimiter=':', usecols=1, dtype=int)

    with rt.DataFile(data_file) as df:
        tadc = df.tadc #rt.TADC(data_file)

        n_du = len( tadc.get_list_of_all_used_dus() )

        traces               = np.zeros( (0,N_CHANNELS,N_SAMPLES),dtype=int )
        du_ids               = np.zeros( (0),dtype=int )
        du_seconds           = np.zeros( (0),dtype=int )
        du_nanoseconds       = np.zeros( (0),dtype=int )

        for entry in mine_event_idcs:
            tadc.get_entry(entry)

            traces         = np.vstack( (traces,np.array(tadc.trace_ch)) )
            du_ids         = np.hstack( (du_ids,np.array(tadc.du_id)) )
            du_seconds     = np.hstack( (du_seconds,np.array(tadc.du_seconds)) )
            du_nanoseconds = np.hstack( (du_nanoseconds,np.array(tadc.du_nanoseconds)) )

    # Delete float channel from the traces
    traces = np.delete(traces,FLOAT_CH,axis=1)

    logger.info(f'Loaded {traces.shape[0]} traces accross {n_du} DUs')

    data = {'traces' : traces,
            'du_ids' : du_ids,
            'du_seconds' : du_seconds,
            'du_nanoseconds' : du_nanoseconds}

    return data


###########################
# DEFINE PARSER ARGUMENTS #
###########################

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Find background pulses in a GP80 MD data file.")

    parser.add_argument('-df',
                        dest='data_file_tag',
                        default=0,
                        type=int,
                        help='Specify the tag of the data file to analyze.')

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

    data = load_data(DATA_FILE,MINE_EVTS_FILE)

    # Perform the FLT-0 trigger algorithm on the filtered traces
    logger.info(f'>>> Performing FLT-0 trigger algorithm in {MODE_FLT0} mode...')
    logger.info(f'FLT-0 parameters: {DICT_TRIGGER_PARAMS}')
    logger.info(f'Channels used for FLT-0 trigger: {np.array(["X","Y","Z"])[CHANNELS_FLT0]}')
    logger.info(f'Samples ignored from beginning of trace: {SAMPLES_FROM_EDGE}')
    res_FLT0 = tools.do_FLT0(data['traces'],
                             DICT_TRIGGER_PARAMS,
                             channels=CHANNELS_FLT0,
                             samples_from_edge=SAMPLES_FROM_EDGE)
    
    # Define a positive trigger as an OR: enough to trigger in any of the desired channels
    if MODE_FLT0 == 'OR':
        mask_FLT0_pass = np.any( res_FLT0['FLT0_flags'][:,CHANNELS_FLT0],axis=1 )
    if MODE_FLT0 == 'AND':
        mask_FLT0_pass = np.all( res_FLT0['FLT0_flags'][:,CHANNELS_FLT0],axis=1 )

    logger.info(f'{np.sum(mask_FLT0_pass)} traces pass FLT-0')
    
    # Compute the SNR for each event
    logger.info(f'>>> Computing SNRs...')
    snr, t_pulse = tools.get_snr_and_t_pulse(data['traces'],
                                             res_FLT0['FLT0_flags'],
                                             res_FLT0['FLT0_first_T1_idcs'],
                                             samples_from_edge=SAMPLES_FROM_EDGE)
    
    logger.info(f'Saving final background database at: {OUT_FILE}')

    np.savez(OUT_FILE,
             traces=data['traces'][mask_FLT0_pass],
             FLT0_flags=res_FLT0['FLT0_flags'][mask_FLT0_pass],
             snr=snr[mask_FLT0_pass],
             t_pulse=t_pulse[mask_FLT0_pass])

    logger.info('*** END OF SCRIPT ***')