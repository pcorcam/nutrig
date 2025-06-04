'''
This scripts creates a database of background pulses for NUTRIG studies.

The following steps are followed:
    1. Take non-filtered GP80 CD data from runs 10083, 10085, 10086.
    2. Filter the data using the offline notch-filter implementation that mimics the notch filters on the FEB.
    3. Identify pulses by applying the offline FLT-0 algorithm that mimics the FLT-0 (or L1) on the FEB.
    4. Store the pulses and trigger information.
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

RUN_NUMBER          = 10086 #10083
DATA_DIR            = '/sps/grand/data/gp80/GrandRoot/2025/05' #/05
DATA_FILES          = sorted( glob.glob( os.path.join( DATA_DIR,f'GP80_*_RUN{RUN_NUMBER}_CD_*.root' ) ) )
FLOAT_CH            = 0
N_CHANNELS          = 4
N_SAMPLES           = 1024
DU_IDS_EXCLUDE      = []
try:
    BASELINE_CALIB = tools.get_baseline_calib(baseline_dir=f'/sps/grand/pcorrea/gp80/baseline_calibration/RUN{RUN_NUMBER}')
except:
    BASELINE_CALIB = None

DICT_TRIGGER_PARAMS = {'th1'      : 55,#45, # [ADC counts]
                       'th2'      : 40,#35, # [ADC counts]
                       't_quiet'  : 500,#500, # [ns]
                       't_period' : 500,#1000, # [ns]
                       't_sepmax' : 20,#200, # [ns]
                       'nc_min'   : 2,
                       'nc_max'   : 7} #10
SAMPLES_FROM_EDGE   = 100 # [ADC samples]
CHANNELS_FLT0       = [0,1] # (0,1,2) = (X,Y,Z) ASSUMING FLOAT CHANNEL IS DISCARDED
MODE_FLT0           = 'OR' # Mode for the FLT-0. Can be 'OR' or 'AND'

CHANNEL_POL       = {0:'X',1:'Y',2:'Z'}
CHANNELS_FLT0_STR = ''
for ch in CHANNELS_FLT0:
    CHANNELS_FLT0_STR += CHANNEL_POL[ch]

OUT_BASEDIR = '/sps/grand/pcorrea/nutrig/database/v2/bkg/'

# For each event, we only want the data of one of the triggered DUs. Set the random generator's seed here
np.random.seed(80) # for GP80!


#############
# FUNCTIONS #
#############

def load_data(data_file,
              event_entry,
              du_ids_exclude=[],
              baseline_calib=None):
    '''
    Load selected data from a GP80 MD file.

    Arguments
    ---------
    - `data_file`
        + type        : `str`
        + description : Absolute path to GP80 MD file in GrandRoot format.

    - `du_ids_exclude`
        + type        : `list`
        + description : List of DU IDs to exclude.

    - `baseline_calib`
        + type        : `np.ndarray[int]`
        + description : Calibrations to apply to the baseline of the traces.

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
    #logger.info(f'Excluding the following DU IDs: {du_ids_exclude}')

    data = {}

    with rt.DataFile(data_file) as df:
        tadc        = df.tadc
        trawvoltage = df.trawvoltage

        tadc.get_entry(event_entry)
        trawvoltage.get_entry(event_entry)

        traces = np.array( tadc.trace_ch )
        du_ids = np.array( tadc.du_id )

        # Delete float channel from the traces
        traces = np.delete(traces,FLOAT_CH,axis=1)

        # Calibrate the mean baseline to 0
        if baseline_calib is not None:
            for i, du_id in enumerate(du_ids):
                try:
                    traces[i] -= baseline_calib[du_id][:,np.newaxis]
                except:
                    continue # this is just for debugging when we don't use the full input data for testing

        # Get the DU coordinates
        gps_long = np.array( trawvoltage.gps_long ) # [deg]
        gps_lat  = np.array( trawvoltage.gps_lat ) # [deg]
        gps_alt  = np.array( trawvoltage.gps_alt ) # [m]
        du_xyz   = tools.get_du_xyz( gps_long,gps_lat,gps_alt ) # [m]

        # There is a bug in event number, so add as follows: file_number*1000 + entry
        file_number  = int( data_file.split('-')[-1].replace('.root','') )
        event_number = int( file_number*1000 + event_entry )

        data['traces']         = traces
        data['du_ids']         = du_ids
        data['du_seconds']     = np.array( tadc.du_seconds )
        data['du_nanoseconds'] = np.array( tadc.du_nanoseconds )
        data['du_xyz']         = du_xyz
        data['event_number']   = event_number
        data['run_number']     = tadc.run_number

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

    try:
        data_file = DATA_FILES[parse_args.data_file_tag]
        logger.info(f'*** Searching for FLT-0 pulses in {data_file} ***')

        logger.info(f'>>> Performing FLT-0 trigger algorithm in {MODE_FLT0} mode...')
        logger.info(f'FLT-0 parameters: {DICT_TRIGGER_PARAMS}')
        logger.info(f'Channels used for FLT-0 trigger: {np.array(["X","Y","Z"])[CHANNELS_FLT0]}')
        logger.info(f'Samples ignored from beginning of trace: {SAMPLES_FROM_EDGE}')

        # First save metadata corresponding to the FLT-0 settings
        metadata_subdir = f'SLT_GP80_RUN_{RUN_NUMBER}_CH_{CHANNELS_FLT0_STR}_MODE_{MODE_FLT0}_TH1_{DICT_TRIGGER_PARAMS["th1"]}_TH2_{DICT_TRIGGER_PARAMS["th2"]}_TQUIET_{DICT_TRIGGER_PARAMS["t_quiet"]}_TPER_{DICT_TRIGGER_PARAMS["t_period"]}_TSEPMAX_{DICT_TRIGGER_PARAMS["t_sepmax"]}_NCMIN_{DICT_TRIGGER_PARAMS["nc_min"]}_NCMAX_{DICT_TRIGGER_PARAMS["nc_max"]}'
        metadata_absdir = os.path.join( OUT_BASEDIR, metadata_subdir )
        metadata_file   = os.path.join( metadata_absdir, 'metadata.npz' )

        if not os.path.exists(metadata_absdir):
          os.makedirs(metadata_absdir)

        logger.info(f'Saving metadata in {metadata_file}...')
        np.savez(metadata_file,
                 dict_trigger_params=DICT_TRIGGER_PARAMS,
                 root_data_dir=DATA_DIR,
                 run_number=RUN_NUMBER,
                 du_ids_exclude=DU_IDS_EXCLUDE,
                 samples_from_edge=SAMPLES_FROM_EDGE,
                 t_eff=-1, # [ns]
                 channel_pol=CHANNEL_POL,
                 channels_flt0=CHANNELS_FLT0,
                 mode_flt0=MODE_FLT0)
        
        # Create subdir for filtered traces
        filtered_absdir = os.path.join( metadata_absdir,'filtered' )
        if not os.path.exists(filtered_absdir):
            os.makedirs(filtered_absdir)

        # Create subdir for raw traces
        raw_absdir = os.path.join( metadata_absdir,'raw' )
        if not os.path.exists(raw_absdir):
            os.makedirs(raw_absdir)

        # Loop over all events in file
        with rt.DataFile(data_file) as df:
            n_events = df.tadc.get_number_of_entries()

        for event_entry in range(n_events)[:]:
            # Load the traces
            logger.info(f'>>> Loading data for event {event_entry+1}/{n_events}...')

            try:
                data = load_data(data_file,event_entry,du_ids_exclude=DU_IDS_EXCLUDE,baseline_calib=BASELINE_CALIB)
            except:
                logger.warning('Data corrupted! Skipping...')
                continue

            # Filter the traces
            logger.info(f'>>> Filtering traces...')
            traces_filtered = tools.filter_traces_bandpass(data['traces']) # you can also choose notch filters here if you prefer

            # Perform the FLT-0 trigger algorithm on the filtered traces
            logger.info(f'>>> Performing FLT-0...')
            res_FLT0 = tools.do_FLT0(traces_filtered,
                                     DICT_TRIGGER_PARAMS,
                                     channels=CHANNELS_FLT0,
                                     samples_from_edge=SAMPLES_FROM_EDGE)
            
            # Define a positive trigger as an OR: enough to trigger in any of the desired channels
            if MODE_FLT0 == 'OR':
                mask_FLT0_pass = np.any( res_FLT0['FLT0_flags'][:,CHANNELS_FLT0],axis=1 )
            if MODE_FLT0 == 'AND':
                mask_FLT0_pass = np.all( res_FLT0['FLT0_flags'][:,CHANNELS_FLT0],axis=1 )

            # Check if there were any triggers
            if not np.any(mask_FLT0_pass):
                logger.warning(f'>>> No FLT-0 triggers for this event! Skipping...')
            else:
                # Compute the SNR for each event, 0 if no FLT-0 trigger
                logger.info(f'>>> Computing SNRs...')
                snr, t_pulse = tools.get_snr_and_t_pulse(traces_filtered,
                                                        res_FLT0['FLT0_flags'],
                                                        res_FLT0['FLT0_first_T1_idcs'],
                                                        samples_from_edge=SAMPLES_FROM_EDGE)
                
                # Save the filtered traces that have passed the FLT-0
                # Also save all related FLT-0 info
                filtered_file = os.path.join( filtered_absdir,'FILTERED_' + os.path.basename(data_file).replace('.root',f'_event_{data["event_number"]}.npz') )
                
                logger.info(f'>>> Saving FILTERED traces that pass FLT-0 in {filtered_file}...')
                logger.info(f'>>> Also saving FLT-0 results in same file...')
                np.savez(filtered_file,
                         traces=traces_filtered,
                         snr=snr,
                         t_pulse=t_pulse,
                         du_ids=data['du_ids'],
                         du_seconds=data['du_seconds'],
                         du_nanoseconds=data['du_nanoseconds'],
                         du_xyz=data['du_xyz'],
                         event_number=data['event_number'],
                         run_number=data['run_number'],
                         FLT0_flags=res_FLT0['FLT0_flags'],
                         FLT0_first_T1_idcs=res_FLT0['FLT0_first_T1_idcs'],
                         n_FLT0=res_FLT0['n_FLT0'])
                
                # Save raw non-filtered traces for completeness
                raw_file = os.path.join( raw_absdir,'RAW_' + os.path.basename(data_file).replace('.root',f'_event_{data["event_number"]}.npz') )

                logger.info(f'Saving corresponding RAW traces in {raw_file}...')
                np.savez(raw_file,
                         traces=data['traces'][mask_FLT0_pass])

    except Exception as e:
        logger.error(f'{e}')

    logger.info('*** END OF SCRIPT ***')