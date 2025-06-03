'''
This scripts creates a database of background pulses for NUTRIG studies.

The following steps are followed:
    1. Take non-filtered GP80 UD data from run 149 (12 March 2025).
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

RUN_NUMBER          = 149 #145
DATA_DIR            = '/sps/grand/data/gp80/GrandRoot/2025/03'
DATA_FILES          = sorted( glob.glob( os.path.join( DATA_DIR,f'GP80_*_RUN{RUN_NUMBER}_UD_*.root' ) ) )
FLOAT_CH            = 0
N_CHANNELS          = 4
N_SAMPLES           = 1024
DU_IDS_EXCLUDE      = []
try:
    BASELINE_CALIB = tools.get_baseline_calib(baseline_dir=f'/sps/grand/pcorrea/gp80/baseline_calibration/RUN{RUN_NUMBER}')
except:
    BASELINE_CALIB = None
    
DICT_TRIGGER_PARAMS = {'th1'      : 45,#45, # [ADC counts]
                       'th2'      : 35,#35, # [ADC counts]
                       't_quiet'  : 500, # [ns]
                       't_period' : 1000, # [ns]
                       't_sepmax' : 200, # [ns]
                       'nc_min'   : 2,
                       'nc_max'   : 10}
SAMPLES_FROM_EDGE   = 100 # [ADC samples]
CHANNELS_FLT0       = [0,1] # (0,1,2) = (X,Y,Z) ASSUMING FLOAT CHANNEL IS DISCARDED
MODE_FLT0           = 'OR' # Mode for the FLT-0. Can be 'OR' or 'AND'

CHANNEL_POL       = {0:'X',1:'Y',2:'Z'}
CHANNELS_FLT0_STR = ''
for ch in CHANNELS_FLT0:
    CHANNELS_FLT0_STR += CHANNEL_POL[ch]

OUT_BASEDIR = '/sps/grand/pcorrea/nutrig/database/v2/bkg/'

# Run 149 was restarted a lot without updating the run number. Only take a slice that we know is stable.
# See also `/pbs/home/p/pcorrea/grand/data_analysis/gp80/plot_rate_UD.py`
if RUN_NUMBER == 149:
    for i, data_file in enumerate(DATA_FILES):
        if os.path.basename(data_file) == 'GP80_20250312_123301_RUN149_UD_RAW-ChanXYZ-20dB-DU33-DU44-DU46-XorY-X2X-Y2Y-0001.root':# 'GP80_20250313_015741_RUN149_UD_RAW-ChanXYZ-20dB-DU33-DU44-DU46-XorY-X2X-Y2Y-0001.root':
            idx_data_file_start = i
        if os.path.basename(data_file) == 'GP80_20250312_171740_RUN149_UD_RAW-ChanXYZ-20dB-DU33-DU44-DU46-XorY-X2X-Y2Y-0370.root': #GP80_20250313_071345_RUN149_UD_RAW-ChanXYZ-20dB-DU33-DU44-DU46-XorY-X2X-Y2Y-1996.root':
            idx_data_file_end = i+1
            break
    print(idx_data_file_start,idx_data_file_end)
    DATA_FILES = DATA_FILES[idx_data_file_start:idx_data_file_end]


#############
# FUNCTIONS #
#############

def load_data(data_file,
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
    logger.info(f'Excluding the following DU IDs: {du_ids_exclude}')

    with rt.DataFile(data_file) as df:
        tadc = df.tadc #rt.TADC(data_file)

        n_entries = tadc.get_number_of_entries()
        n_du      = len( tadc.get_list_of_all_used_dus() )

        traces               = np.zeros( (n_entries,N_CHANNELS,N_SAMPLES),dtype=int )
        du_ids               = np.zeros( (n_entries),dtype=int )
        du_seconds           = np.zeros( (n_entries),dtype=int )
        du_nanoseconds       = np.zeros( (n_entries),dtype=int )
        mask_du_exclude      = np.ones( (n_entries),dtype=bool )

        for entry in range(n_entries):
            tadc.get_entry(entry)

            if tadc.du_id[0] in du_ids_exclude:
                mask_du_exclude[entry] = False
                continue

            # Only one DU per entry
            traces[entry]         = tadc.trace_ch[0]
            du_ids[entry]         = tadc.du_id[0]
            du_seconds[entry]     = tadc.du_seconds[0]
            du_nanoseconds[entry] = tadc.du_nanoseconds[0]

    # Remove entries for DUs we exclude
    traces         = traces[mask_du_exclude]
    du_ids         = du_ids[mask_du_exclude]
    du_seconds     = du_seconds[mask_du_exclude]
    du_nanoseconds = du_nanoseconds[mask_du_exclude]

    # Delete float channel from the traces
    traces = np.delete(traces,FLOAT_CH,axis=1)

    # Calibrate the mean baseline to 0
    if baseline_calib is not None:
        for i, du_id in enumerate(du_ids):
            try:
                traces[i] -= baseline_calib[du_id][:,np.newaxis]
            except:
                continue # this is just for debugging when we don't use the full input data for testing

    logger.info(f'Loaded {n_entries} traces accross {n_du - len(du_ids_exclude)} DUs')

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

    try:
        data_file = DATA_FILES[parse_args.data_file_tag]
        logger.info(f'*** Searching for FLT-0 pulses in {data_file} ***')

        # Load the traces
        logger.info(f'>>> Loading data...')
        data = load_data(data_file,
                         du_ids_exclude=DU_IDS_EXCLUDE,
                         baseline_calib=BASELINE_CALIB)

        # Filter the traces
        logger.info(f'>>> Filtering traces...')
        traces_filtered = tools.filter_traces_bandpass(data['traces']) # you can also choose notch filters here if you prefer

        # Perform the FLT-0 trigger algorithm on the filtered traces
        logger.info(f'>>> Performing FLT-0 trigger algorithm in {MODE_FLT0} mode...')
        logger.info(f'FLT-0 parameters: {DICT_TRIGGER_PARAMS}')
        logger.info(f'Channels used for FLT-0 trigger: {np.array(["X","Y","Z"])[CHANNELS_FLT0]}')
        logger.info(f'Samples ignored from beginning of trace: {SAMPLES_FROM_EDGE}')
        res_FLT0 = tools.do_FLT0(traces_filtered,
                                 DICT_TRIGGER_PARAMS,
                                 channels=CHANNELS_FLT0,
                                 samples_from_edge=SAMPLES_FROM_EDGE)
        
        # Define a positive trigger as an OR: enough to trigger in any of the desired channels
        if MODE_FLT0 == 'OR':
          mask_FLT0_pass = np.any( res_FLT0['FLT0_flags'][:,CHANNELS_FLT0],axis=1 )
        if MODE_FLT0 == 'AND':
          mask_FLT0_pass = np.all( res_FLT0['FLT0_flags'][:,CHANNELS_FLT0],axis=1 )
        
        # Compute the average trigger rate for each DU
        logger.info(f'>>> Computing FLT-0 trigger rates...')
        res_FLT0_trigger_rate = tools.compute_FLT0_trigger_rate_UD(res_FLT0['n_FLT0'],
                                                                   res_FLT0['FLT0_flags'],
                                                                   data['du_ids'],
                                                                   data['du_seconds'],
                                                                   data['du_nanoseconds'],
                                                                   channels=CHANNELS_FLT0)
        
        # Compute the SNR for each event
        logger.info(f'>>> Computing SNRs...')
        snr, t_pulse = tools.get_snr_and_t_pulse(traces_filtered,
                                                 res_FLT0['FLT0_flags'],
                                                 res_FLT0['FLT0_first_T1_idcs'],
                                                 samples_from_edge=SAMPLES_FROM_EDGE)

        # First save metadata corresponding to the FLT-0 settings
        metadata_subdir = f'GP80_RUN_{RUN_NUMBER}_CH_{CHANNELS_FLT0_STR}_MODE_{MODE_FLT0}_TH1_{DICT_TRIGGER_PARAMS["th1"]}_TH2_{DICT_TRIGGER_PARAMS["th2"]}_TQUIET_{DICT_TRIGGER_PARAMS["t_quiet"]}_TPER_{DICT_TRIGGER_PARAMS["t_period"]}_TSEPMAX_{DICT_TRIGGER_PARAMS["t_sepmax"]}_NCMIN_{DICT_TRIGGER_PARAMS["nc_min"]}_NCMAX_{DICT_TRIGGER_PARAMS["nc_max"]}'
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
        
        # Save the filtered traces that have passed the FLT-0
        # Also save all related FLT-0 info
        filtered_absdir = os.path.join( metadata_absdir,'filtered' )
        filtered_file   = os.path.join( filtered_absdir,'FILTERED_' + os.path.basename(data_file).replace('.root','.npz') )
        
        if not os.path.exists(filtered_absdir):
            os.makedirs(filtered_absdir)

        logger.info(f'Saving FILTERED traces that pass FLT-0 in {filtered_file}...')
        logger.info(f'Also saving FLT-0 results and trigger rates in same file...')
        np.savez(filtered_file,
                 traces=traces_filtered[mask_FLT0_pass],
                 snr=snr[mask_FLT0_pass],
                 t_pulse=t_pulse[mask_FLT0_pass],
                 du_ids=data['du_ids'][mask_FLT0_pass],
                 du_seconds=data['du_seconds'][mask_FLT0_pass],
                 du_nanoseconds=data['du_nanoseconds'][mask_FLT0_pass],
                 FLT0_flags=res_FLT0['FLT0_flags'][mask_FLT0_pass],
                 FLT0_first_T1_idcs=res_FLT0['FLT0_first_T1_idcs'][mask_FLT0_pass],
                 n_FLT0=res_FLT0['n_FLT0'][mask_FLT0_pass],
                 trigger_rate_per_ch=res_FLT0_trigger_rate['trigger_rate_per_ch'],
                 trigger_rate_OR=res_FLT0_trigger_rate['trigger_rate_OR'],
                 trigger_rate_AND=res_FLT0_trigger_rate['trigger_rate_AND'],
                 t_bin_trigger_rate=res_FLT0_trigger_rate['t_bin'])
        
        # Save raw non-filtered traces for completeness
        raw_absdir = os.path.join( metadata_absdir,'raw' )
        raw_file   = os.path.join( raw_absdir,'RAW_' + os.path.basename(data_file).replace('.root','.npz') )
        
        if not os.path.exists(raw_absdir):
            os.makedirs(raw_absdir)

        logger.info(f'Saving corresponding RAW traces in {raw_file}...')
        np.savez(raw_file,
                 traces=data['traces'][mask_FLT0_pass])

    except Exception as e:
        logger.error(f'{e}')

    logger.info('*** END OF SCRIPT ***')