'''
This scripts creates a preselection of templates for the NUTRIG template-fit method.
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

import template_lib.tools as tools

from database.tools import filter_traces_bandpass

logger = logging.getLogger(__name__)


####################
# FIXED PARAMETERS #
####################

SIM_SOFTWARE          = 'ZHAireS'
SIM_TAG               = 'DC2.1rc2'
SIM_BASEDIR           = f'/sps/grand/{SIM_TAG}'
SIM_ADC_FILES_NJ      = sorted( glob.glob( os.path.join( SIM_BASEDIR,f'sim_*_{SIM_SOFTWARE}-NJ_*/**/adc_*_L1_*.root' ), recursive=True) )
SIM_SHOWER_FILES      = sorted( glob.glob( os.path.join( SIM_BASEDIR,f'sim_*_{SIM_SOFTWARE}-NJ_*/**/shower_*_L0_*.root' ), recursive=True) )
SIM_RUN_FILE          = sorted( glob.glob( os.path.join( SIM_BASEDIR,f'sim_*_{SIM_SOFTWARE}-NJ_*/**/run_*_L1_*.root' ), recursive=True) )[0]
FILTER_TAG            = 'FILTERED'

THRESH   = 100 # ADC counts
CHANNELS = [0,1]
POL      = 'XY'

BEFORE_PEAK = 30
AFTER_PEAK  = 70
N_SAMPLES   = BEFORE_PEAK+AFTER_PEAK

OUT_BASEDIR = '/sps/grand/pcorrea/nutrig/template/v2/preselection'
OUT_DIR     = os.path.join(OUT_BASEDIR,f'{SIM_SOFTWARE}_{SIM_TAG}_{FILTER_TAG}_{POL}_THRESH_{THRESH}')


#############
# FUNCTIONS #
#############

def where_above_thresh(traces,
                       thresh,
                       channels=[0,1]):
    '''
    Finds which traces have a maximum value above a certain threshold.
    Multiple channels can be considered simultaneously.

    Arguments
    ---------
    - `traces`
        + type        : `np.ndarray[int]`
        + units       : ADC counts
        + description : ADC traces, with shape `(N_entries,N_channels,N_samples)`.

    - `thresh`
        + type        : `int`
        + units       : ADC counts
        + description : Required threshold for the traces to exceed.

    - `channels`
        + type        : `list[int]`
        + description : Channels to check for the threshold requirement. Can only contain values in [0,1,2].

    Returns
    -------
    - `where_above_thresh`
        + type        : `np.ndarray[bool]`
        + description : Flattened mask of the traces that exceed the threshold, with shape `(N_entries,)`.
    '''
    
    n_entries  = traces.shape[0]
    n_channels = traces.shape[1]

    if not np.all( np.isin( channels,range(n_channels) ) ):
        raise Exception(f'Channels needs to be a list with [0,1,2]')

    max_traces            = np.max( np.abs(traces),axis=2 )
    where_above_thresh    = np.zeros(n_entries*n_channels,dtype=bool)

    for ch in range(n_channels):
        if ch in channels:
            # This will ensure ordering X,Y,Z,X,Y,Z,...
            where_above_thresh[ch::n_channels] = np.where(max_traces[:,ch]>thresh,True,False)
    
    return where_above_thresh


def get_traces_around_peaks(traces,
                            before_peak,
                            after_peak):
    '''
    Only keeps a selected amount of samples of the traces around the pulse peak.
    Coded with help of ChatGPT.
    '''

    # Step 1: Find peak indices per trace
    peaks = np.argmax(np.abs(traces), axis=2) # shape (100, 3)

    # Step 2: Build index grid for window extraction
    # Shape: (100, 3, 100)
    window_offsets = np.arange(-before_peak, after_peak)[None, None, :]  # shape (1, 1, 100)
    peak_indices   = peaks[:, :, None]  # shape (100, 3, 1)
    sample_indices = peak_indices + window_offsets  # shape (100, 3, 100)

    # Step 3: Gather the samples using fancy indexing
    # We need to prepare broadcastable trace and channel indices
    trace_idx = np.arange(traces.shape[0])[:, None, None]  # shape (100, 1, 1)
    chan_idx  = np.arange(traces.shape[1])[None, :, None] # shape (1, 3, 1)

    # Final extraction
    traces_cut = traces[trace_idx, chan_idx, sample_indices]  # shape (100, 3, 100)

    return traces_cut


def get_template_preselection(adc_file,
                              shower_file,
                              run_file,
                              thresh,
                              channels=[0,1],
                              filter_tag='FILTERED'):
    
    data = {}
    
    data['templates']    = np.zeros( (0,N_SAMPLES) )
    data['event_number'] = np.zeros( (0,),dtype=int )
    data['run_number']   = np.zeros( (0,),dtype=int )
    data['du_ids']       = np.zeros( (0,),dtype=int )
    data['channels']     = np.zeros( (0,),dtype=int )
    data['energy']       = np.zeros( (0,) )
    data['zenith']       = np.zeros( (0,) )
    data['azimuth']      = np.zeros( (0,) )
    data['omega']        = np.zeros( (0,) )
    data['omega_c']      = np.zeros( (0,) )

    with rt.DataFile(adc_file) as df_adc, rt.DataFile(shower_file) as df_shower, rt.DataFile(run_file) as df_run:
        tadc    = df_adc.tadc
        tshower = df_shower.tshower
        trun    = df_run.trun
        trun.get_entry(0)

        du_xyz  = np.array(trun.du_xyz)

        n_entries = tadc.get_number_of_entries()
        for entry in range(n_entries):
            logger.info(f'Processing entry {entry+1}/{n_entries}...')
            tadc.get_entry(entry), tshower.get_entry(entry)

            traces = np.array( tadc.trace_ch,dtype=float )
            if filter_tag == 'FILTERED':
                traces = filter_traces_bandpass(traces,do_minimum_phase=True)
            if filter_tag == 'RAW':
                pass
            else:
                raise ValueError('Filter tag should be either "FILTERED" or "RAW"')

            # Compute opening angle w.r.t. shower axis and Cherenkov angle
            xmax_pos_shc    = np.array(tshower.xmax_pos_shc)
            shower_core_pos = np.array(tshower.shower_core_pos)
            omega           = np.rad2deg( tools.get_omega_from_Xmax_pos( du_xyz[tadc.du_id],shower_core_pos,xmax_pos_shc ) ) # [deg]
            omega_c         = np.rad2deg( tools.get_omega_c( tools.get_refraction_index_at_pos( xmax_pos_shc ) ) ) # [deg]
            energy          = np.array(tshower.energy_primary)
            zenith          = np.array(tshower.zenith)
            azimuth         = np.array(tshower.azimuth)

            # Add correct dimensions for flattened array
            du_ids   = np.repeat( tadc.du_id,traces.shape[1] )
            channels = np.tile( range( traces.shape[1] ),traces.shape[0] )
            omega    = np.repeat( omega,traces.shape[1] )


            # Find which traces are above the required threshold
            mask_above_thresh = where_above_thresh(traces,thresh,channels=channels)

            # Get the samples of interest around the trace peaks for the templates
            traces_cut = get_traces_around_peaks(traces,BEFORE_PEAK,AFTER_PEAK)

            # Flatten along polarization axis; order is now X,Y,Z,X,Y,Z...
            traces_flattened = traces_cut.reshape(-1,N_SAMPLES)

            # Only select the templates that exceeded the required threshold
            templates_entry  = traces_flattened[mask_above_thresh]

            # Normalize the templates
            templates_entry /= np.linalg.norm(templates_entry,axis=1,keepdims=True)

            data['templates']    = np.concatenate( ( data['templates'],templates_entry ) )
            data['event_number'] = np.concatenate( ( data['event_number'],tadc.event_number*np.ones( templates_entry.shape[0],dtype=int ) ) )
            data['run_number']   = np.concatenate( ( data['run_number'],tadc.run_number*np.ones( templates_entry.shape[0],dtype=int ) ) )
            data['du_ids']       = np.concatenate( ( data['du_ids'],du_ids[mask_above_thresh ] ) )
            data['channels']     = np.concatenate( ( data['channels'],channels[mask_above_thresh ] ) )
            data['omega']        = np.concatenate( ( data['omega'],omega[mask_above_thresh ] ) )
            data['omega_c']      = np.concatenate( ( data['omega_c'],omega_c*np.ones( templates_entry.shape[0] ) ) )
            data['energy']       = np.concatenate( ( data['energy'],energy*np.ones( templates_entry.shape[0] ) ) )
            data['zenith']       = np.concatenate( ( data['zenith'],zenith*np.ones( templates_entry.shape[0] ) ) )
            data['azimuth']      = np.concatenate( ( data['azimuth'],azimuth*np.ones( templates_entry.shape[0] ) ) )

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

    # Get the matching ADC and shower files
    # NJ -> no added noise for templates
    adc_file    = SIM_ADC_FILES_NJ[parse_args.data_file_tag]
    shower_file = SIM_SHOWER_FILES[parse_args.data_file_tag]

    logger.info(f'*** Making preselection of templates from {adc_file} ***')
    logger.info(f'*** Corresponding TShower file: {shower_file} ***')
    logger.info(f'*** Corresponding TRun file: {SIM_RUN_FILE} ***')
    logger.info(f'*** Required ADC threshold for selection: {THRESH} ***')
    logger.info(f'*** Polarization: {POL} ***')

    data = get_template_preselection(adc_file,
                                     shower_file,
                                     SIM_RUN_FILE,
                                     THRESH,
                                     channels=CHANNELS)
    
    logger.info(f'Preselected {data["templates"].shape[0]} templates!')

    # Create subdir for filtered traces
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    file_name = f'preselection_templates_{POL}_'+os.path.basename(adc_file).replace('.root','.npz')
    out_file  = os.path.join(OUT_DIR,'preselection_templates_'+os.path.splitext( os.path.basename(adc_file) )[0]+'.npz')

    logger.info(f'Saving preselected templates in {out_file}')
    np.savez(out_file,
             templates=data['templates'],
             event_number=data['event_number'],
             run_number=data['run_number'],
             du_ids=data['du_ids'],
             channels=data['channels'],
             omega=data['omega'],
             omega_c=data['omega_c'],
             energy=data['energy'],
             zenith=data['zenith'],
             azimuth=data['azimuth'])
    
    logger.info('*** END OF SCRIPT ***')