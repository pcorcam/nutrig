'''
This scripts creates a database of air-shower simulations for NUTRIG studies.

The following steps are followed:
    1. Take DC2 simulations of ZHAireS / CoREAS. AN uses "static noise" traces of GP80 MD data from run 145 (4 February 2025).
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

from template_lib.tools import get_refraction_index_at_pos, get_omega_c, get_omega_from_Xmax_pos

logger = logging.getLogger(__name__)


####################
# FIXED PARAMETERS #
####################

ARRAY                 = 'GP289'
SIM_SOFTWARE          = 'ZHAireS'
SIM_TAG               = 'DC2.1rc4'
SIM_BASEDIR           = f'/sps/grand/{SIM_TAG}/{ARRAY}{SIM_SOFTWARE}-AN'
SIM_ADC_FILES         = sorted( glob.glob( os.path.join( SIM_BASEDIR,f'sim_*_{ARRAY}{SIM_SOFTWARE}-AN_*/**/adc_*_L1_*.root' ), recursive=True ) )
SIM_SHOWER_FILES      = sorted( glob.glob( os.path.join( SIM_BASEDIR,f'sim_*_{ARRAY}{SIM_SOFTWARE}-AN_*/**/shower_*_L0_*.root' ), recursive=True ) )
SIM_RUN_FILES         = sorted( glob.glob( os.path.join( SIM_BASEDIR,f'sim_*_{ARRAY}{SIM_SOFTWARE}-AN_*/**/run_*_L1_*.root' ), recursive=True ) )
N_EVENTS_FILE         = 1000
SAMPLE_PULSE_PEAK_INJ = int(550/2)
JITTER_WIDTH          = 20

# N_CHANNELS          = 3
# N_SAMPLES           = 1024

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

OUT_BASEDIR = '/sps/grand/pcorrea/nutrig/database/v2/sig/'

APPLY_FILTER = True


#############
# FUNCTIONS #
#############

def load_data(adc_file,
              shower_file,
              run_file,
              event_entry):
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

    data = {}

    # Get data from ADC file
    with rt.DataFile(adc_file) as df_adc:
        tadc = df_adc.tadc
        tadc.get_entry(event_entry)

        data['traces']         = np.array( tadc.trace_ch )
        data['du_ids']         = np.array( tadc.du_id )
        data['du_seconds']     = np.array( tadc.du_seconds )
        data['du_nanoseconds'] = np.array( tadc.du_nanoseconds )
        data['event_number']   = tadc.event_number
        data['run_number']     = tadc.run_number

    # Get data from run file
    with rt.DataFile(run_file) as df_run:
        trun = df_run.trun
        trun.get_entry(0)

        data['du_xyz'] = np.array( trun.du_xyz )[data['du_ids']]

    # Get data from shower file
    with rt.DataFile(shower_file) as df_shower:
        tshower = df_shower.tshower
        tshower.get_entry(event_entry)

        data['primary_type']    = tshower.primary_type
        data['energy_primary']  = tshower.energy_primary
        data['zenith']          = tshower.zenith
        data['azimuth']         = tshower.azimuth
        data['shower_core_pos'] = np.array(tshower.shower_core_pos)
        data['xmax_pos_shc']    = np.array(tshower.xmax_pos_shc)
        data['omega']           = np.rad2deg( get_omega_from_Xmax_pos(data['du_xyz'],data['shower_core_pos'],data['xmax_pos_shc']) ) # [deg]
        data['omega_c']         = np.rad2deg( get_omega_c( get_refraction_index_at_pos(data['xmax_pos_shc']) ) ) # [deg]

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

    logger.debug(SIM_ADC_FILES)
    logger.debug(os.path.join( SIM_BASEDIR,f'sim_*_{ARRAY}{SIM_SOFTWARE}-AN_*/**/adc_*_L1_*.root' ))

    try:
        # Get the matching ADC and shower files
        adc_file    = SIM_ADC_FILES[parse_args.data_file_tag]
        shower_file = SIM_SHOWER_FILES[parse_args.data_file_tag]

        # Find the corresponding run file
        run_str  = adc_file.split('/')[-2].split('_')[4].replace('RUN','run_')
        run_file = None

        for sim_run_file in SIM_RUN_FILES:
            if run_str in os.path.basename(sim_run_file):
                run_file = sim_run_file
                break
            
        if run_file is None:
            raise Exception(f'No run file for chosen ADC file: {adc_file}')


        logger.info(f'*** Searching for events with FLT-0 pulses in {adc_file} ***')
        logger.info(f'*** Corresponding TShower file: {shower_file} ***')
        logger.info(f'*** Corresponding TRun file: {run_file} ***')
        
        logger.info(f'*** Performing FLT-0 trigger algorithm in {MODE_FLT0} mode ***')
        logger.info(f'*** FLT-0 parameters: {DICT_TRIGGER_PARAMS} ***')
        logger.info(f'*** Channels used for FLT-0 trigger: {np.array(["X","Y","Z"])[CHANNELS_FLT0]} ***')
        logger.info(f'*** Samples ignored from beginning of trace: {SAMPLES_FROM_EDGE} ***')


        # First save metadata corresponding to the FLT-0 settings
        metadata_subdir = f'{SIM_SOFTWARE}_{SIM_TAG}_CH_{CHANNELS_FLT0_STR}_MODE_{MODE_FLT0}_TH1_{DICT_TRIGGER_PARAMS["th1"]}_TH2_{DICT_TRIGGER_PARAMS["th2"]}_TQUIET_{DICT_TRIGGER_PARAMS["t_quiet"]}_TPER_{DICT_TRIGGER_PARAMS["t_period"]}_TSEPMAX_{DICT_TRIGGER_PARAMS["t_sepmax"]}_NCMIN_{DICT_TRIGGER_PARAMS["nc_min"]}_NCMAX_{DICT_TRIGGER_PARAMS["nc_max"]}'
        metadata_absdir = os.path.join( OUT_BASEDIR, metadata_subdir )
        metadata_file   = os.path.join( metadata_absdir, 'metadata.npz' )

        if not os.path.exists(metadata_absdir):
          os.makedirs(metadata_absdir)

        logger.info(f'Saving metadata in {metadata_file}...')
        np.savez(metadata_file,
                 dict_trigger_params=DICT_TRIGGER_PARAMS,
                 root_sim_dir=SIM_BASEDIR,
                 sim_software=SIM_SOFTWARE,
                 samples_from_edge=SAMPLES_FROM_EDGE,
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
        for event_entry in range(N_EVENTS_FILE)[:]:

            # Load the traces
            logger.info(f'>>> Loading data for event {event_entry+1}/{N_EVENTS_FILE}...')

            try:
                data = load_data(adc_file,shower_file,run_file,event_entry)
            except:
                logger.warning('Data corrupted! Skipping...')
                continue

            # PROCESS FILTERED TRACES
            
            logger.info('*** FILTERED TRACES ***')

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
                filtered_file = os.path.join( filtered_absdir,'FILTERED_' + os.path.basename(adc_file).replace('.root',f'_run_{data["run_number"]}_event_{data["event_number"]}.npz') )
                
                logger.info(f'>>> Saving FILTERED traces that pass FLT-0 in {filtered_file}...')
                logger.info(f'>>> Also saving FLT-0 results in same file...')
                np.savez(filtered_file,
                        traces=traces_filtered,
                        snr=snr,
                        t_pulse=t_pulse,
                        du_ids=data['du_ids'],
                        du_seconds=data['du_seconds'],
                        du_nanoseconds=data['du_nanoseconds'],
                        event_number=data['event_number'],
                        run_number=data['run_number'],
                        du_xyz=data['du_xyz'],
                        primary_type=data['primary_type'],
                        energy_primary=data['energy_primary'],
                        zenith=data['zenith'],
                        azimuth=data['azimuth'],
                        omega=data['omega'],
                        omega_c=data['omega_c'],
                        shower_core_pos=data['shower_core_pos'],
                        xmax_pos_shc=data['xmax_pos_shc'],
                        FLT0_flags=res_FLT0['FLT0_flags'],
                        FLT0_first_T1_idcs=res_FLT0['FLT0_first_T1_idcs'],
                        n_FLT0=res_FLT0['n_FLT0'])
            
            # PROCESS RAW NON-FILTERED TRACES
            logger.info('*** RAW TRACES ***')

            # Perform the FLT-0 trigger algorithm on the filtered traces
            logger.info(f'>>> Performing FLT-0...')
            res_FLT0 = tools.do_FLT0(data['traces'],
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
                snr, t_pulse = tools.get_snr_and_t_pulse(data['traces'],
                                                        res_FLT0['FLT0_flags'],
                                                        res_FLT0['FLT0_first_T1_idcs'],
                                                        samples_from_edge=SAMPLES_FROM_EDGE)

                # Save the raw non-filtered traces that have passed the FLT-0
                # Also save all related FLT-0 info
                raw_file   = os.path.join( raw_absdir,'RAW_' + os.path.basename(adc_file).replace('.root',f'_run_{data["run_number"]}_event_{data["event_number"]}.npz') )

                logger.info(f'Saving corresponding RAW traces in {raw_file}...')
                np.savez(raw_file,
                        traces=data['traces'],
                        snr=snr,
                        t_pulse=t_pulse,
                        du_ids=data['du_ids'],
                        du_seconds=data['du_seconds'],
                        du_nanoseconds=data['du_nanoseconds'],
                        event_number=data['event_number'],
                        run_number=data['run_number'],
                        du_xyz=data['du_xyz'],
                        primary_type=data['primary_type'],
                        energy_primary=data['energy_primary'],
                        zenith=data['zenith'],
                        azimuth=data['azimuth'],
                        omega=data['omega'],
                        omega_c=data['omega_c'],
                        shower_core_pos=data['shower_core_pos'],
                        xmax_pos_shc=data['xmax_pos_shc'],
                        FLT0_flags=res_FLT0['FLT0_flags'],
                        FLT0_first_T1_idcs=res_FLT0['FLT0_first_T1_idcs'],
                        n_FLT0=res_FLT0['n_FLT0'])
      

    except Exception as e:
        logger.error(f'{e}')


    logger.info('*** END OF SCRIPT ***')