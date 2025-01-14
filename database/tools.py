###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import glob
import sys
import logging

import numpy as np
from scipy import signal

import grand.dataio.root_trees as rt # type: ignore

from simu_analysis.tools import get_refraction_index_at_pos, get_omega_c, get_omega_from_Xmax_pos # type: ignore

logger = logging.getLogger(__name__)


###-###-###-###-###-###-###- LOGGER -###-###-###-###-###-###-###

def get_logging_level(level_str):

    if level_str == 'debug':
        level = logging.DEBUG
    elif level_str == 'info':
        level = logging.INFO
    elif level_str == 'warning':
        level = logging.WARNING
    elif level_str == 'error':
        level = logging.ERROR
    elif level_str == 'critical':
        level = logging.CRITICAL
    else:
        raise Exception('No valid logging level!')
    
    return level


def load_logger(level='info'):
    logger  = logging.getLogger(__name__)
    level   = get_logging_level(level)
    logger.setLevel(level)

    handler   = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s")

    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def digitize(voltage_traces,
             simu_sampling_rate=2e3,
             adc_sampling_rate=500,
             adc_to_voltage=0.9e6/2**13,
             float_adc=False):
    '''
    Description
    -----------
    Performs the virtual digitization of voltage traces at the ADC level:
    - desamples the simulated signal to the ADC sampling rate;
    - converts voltage to ADC counts.
    NOTE: this step should already be included in a future grandlib version!

    Arguments
    ---------
    `voltage_traces`
    type        : np.ndarray[float]
    units       : µV
    description : Array of voltage traces at the ADC level with dimensions (N_du,3,N_simu_samples).

    `simu_sampling_rate`
    type        : float
    units       : MHz
    description : Sampling rate used in the ZHaiRES/CoREAS simulation. Typically 2 GHz.

    `adc_sampling_rate`
    type        : float
    units       : MHz
    description : Sampling rate of the ADC. Currently 500 MHz.

    `adc_to_voltage`
    type        : float
    units       : µV
    description : Conversion factor from ADC counts to voltage.

    `float_adc`
    type        : bool
    description : Option to save ADC counts as a float number, without quantifying to an integer value.
    
                                
    Returns
    -------
    `adc_traces`
    type        : np.ndarray[float]
    units       : LSB
    description : The digitized array of voltage traces, with the ADC sampling rate and in ADC counts.
    '''
    
    # Convert voltage to ADC
    adc_traces = voltage_traces/adc_to_voltage

    # Truncate to get the closest integer
    if not float_adc:
        adc_traces = np.trunc(adc_traces)

    # Obtain desampling factor
    desampling_factor = int(simu_sampling_rate/adc_sampling_rate)

    if desampling_factor < 1:
        raise Exception('Simulation sampling rate can not be lower than ADC sampling rate!',desampling_factor)
    if not ( (desampling_factor & (desampling_factor-1) == 0) and desampling_factor != 0 ):
        warn = 'Desampling factor is not a power of 2: {}'.format(desampling_factor)
        logger.warning(warn)

    # Return trace in ADC counts and with the ADC sampling rate
    return adc_traces[...,::desampling_factor]


def set_trace_length(traces,
                     target_length):
    
    logger.info(f'Setting trace length to {target_length}')

    in_trace_length = traces.shape[-1]
    
    # Pad with zeros at end if trace length is smaller target length
    if in_trace_length <= target_length:
        out_traces = np.zeros( (traces.shape[0],traces.shape[1],target_length) )
        out_traces[...,:in_trace_length] = traces

    # Pulse simulations are normally at the beginning of trace
    else:
        out_traces = traces[...,:target_length]
    
    return out_traces


def extract_shower_params(voltage_file,
                          do_digitization=True):

    try:
        tvoltage = rt.TVoltage(voltage_file)
    except:
        logger.error(f'No valid voltage file: {voltage_file}')
        raise Exception(f'No valid voltage file: {voltage_file}')
        
    sim_dir     = os.path.dirname( os.path.dirname( os.path.abspath(voltage_file) ) )
    efield_dir  = os.path.join( sim_dir,'efield' )
    efield_file = os.path.join( efield_dir,os.path.basename(voltage_file).replace('voltage_','') )

    try:
        trun    = rt.TRun(efield_file)
        tshower = rt.TShower(efield_file)
    except:
        logger.error(f'No valid efield file: {efield_file}')
        raise Exception(f'No valid efield file: {efield_file}')
    
    logger.info(f'Getting voltage traces from {voltage_file}')

    tvoltage.get_entry(0), trun.get_entry(0), tshower.get_entry(0)


    shower_params = {}

    traces = np.array(tvoltage.trace)

    if do_digitization:
        logger.info(f'Digitizing voltage traces to obtain ADC traces')
        traces = digitize(traces)
        traces = traces.astype(int)

    shower_params['traces']          = traces
    shower_params['shower_core_pos'] = np.array(tshower.shower_core_pos)
    shower_params['xmax_pos_shc']    = np.array(tshower.xmax_pos_shc)
    shower_params['du_xyz']          = np.array(trun.du_xyz) # GP300 coords in shower-core frame
    shower_params['du_seconds']      = np.array(tvoltage.du_seconds)
    shower_params['du_nanoseconds']  = np.array(tvoltage.du_nanoseconds)
    shower_params['omega']           = np.rad2deg( get_omega_from_Xmax_pos(shower_params['du_xyz'],shower_params['xmax_pos_shc']) ) # [deg]
    shower_params['omega_c']         = np.rad2deg( get_omega_c( get_refraction_index_at_pos(shower_params['xmax_pos_shc']) ) ) # [deg]
    shower_params['energy']          = tshower.energy_primary # [EeV]
    shower_params['zenith']          = tshower.zenith # [deg]
    shower_params['azimuth']         = tshower.azimuth # [deg]
    shower_params['primary_type']    = tshower.primary_type

    # Put in DU positions in true GP300 coordinates
    shower_params['du_xyz'] += shower_params['shower_core_pos']

    tvoltage.stop_using()
    trun.stop_using()
    tshower.stop_using()

    tvoltage.close_file()
    trun.close_file()
    tshower.close_file()

    return shower_params


def get_du_ids(du_xyz_input,
               gp300_layout_file='/pbs/home/p/pcorrea/grand/layout/F06970G5G2X_GP300_layout_grandcs_DUonGround.dat'):
    '''
    Finds the DU IDs for a set of DU coordinates in the GP300 array.

    Arguments
    ---------
    `du_xyz_input`
    type        : np.ndarray[float]
    units       : m
    description : Array of XYZ positions of a set of DUs, with shape (N_du,3). (X,Y)=(0,0) should be at the center of GP300.

    `gp300_layout_file`
    type        : str
    description : Path to .dat file containing the layout information of GP300.
    
                                
    Returns
    -------
    `du_ids`
    type        : np.ndarray[int]
    description : The DU IDs corresponding to the input coordinates.
    '''

    try:
        gp300_layout_info = np.loadtxt(gp300_layout_file,skiprows=2,usecols=(0,2,3))
    except:
        #logger.error('No valid GP300 layout file provided.')
        raise Exception('No valid GP300 layout file provided.')

    # We only care about the X,Y coordinates here
    gp300_du_ids      = gp300_layout_info[:,0]
    gp300_du_xy       = gp300_layout_info[:,1:]
    du_xy_input       = du_xyz_input[:,:2]

    du_ids = np.zeros(du_xyz_input.shape[0],dtype=int)

    for i, du_pos in enumerate( du_xy_input ):
        for gp300_du_id, gp300_pos in zip(gp300_du_ids,gp300_du_xy):
            if np.linalg.norm( du_pos - gp300_pos ) < 0.1: # to account for numerical errors
                du_ids[i] = gp300_du_id
                break
    
    return du_ids


def extract_trigger_parameters(trace, trigger_config, baseline=0):
    # Extract the trigger infos from a trace

    # Parameters :
    # ------------
    # trace, numpy.ndarray: 
    # traces in ADC unit
    # trigger_config, dict:
    # the trigger parameters set in DAQ

    # Returns :
    # ---------
    # Index in the trace when the first T1 crossing happens
    # Indices in the trace of T2 crossing happens
    # Number of T2 crossings
    # Q, Peak/NC

    # Find the position of the first T1 crossing
    index_t1_crossing = np.where(np.abs(trace) > trigger_config["th1"],
                                 np.arange(len(trace)), -1)
    dict_trigger_infos = dict()
    
    mask_T1_crossing = (index_t1_crossing != -1)
    if sum(mask_T1_crossing) == 0:
        # No T1 crossing 
        raise ValueError("No T1 crossing!")
    
    dict_trigger_infos['index_T1_crossing'] = None
    # Tquiet to decide the quiet time before the T1 crossing 
    for i in index_t1_crossing[mask_T1_crossing]:
       # Abs value not exceeds the T1 threshold
       if i - trigger_config["t_quiet"]//2 < 0:
          raise ValueError("Not enough data before T1 crossing!")
       if np.all(np.abs(trace[np.max(0, i - trigger_config['t_quiet'] // 2):i]) < trigger_config["th1"]):
          dict_trigger_infos["index_T1_crossing"] = i
          # the first T1 crossing satisfying the quiet condition
          break
    if dict_trigger_infos['index_T1_crossing'] == None:
       raise ValueError("No T1 crossing with Tquiet satified!")
    # The trigger logic works for the timewindow given by T_period after T1 crossing.
    # Count number of T2 crossings, relevant pars: T2, NCmin, NCmax, T_sepmax
    # From ns to index, divided by two for 500MHz sampling rate
    period_after_T1_crossing = trace[dict_trigger_infos["index_T1_crossing"]:dict_trigger_infos["index_T1_crossing"]+trigger_config['t_period']//2]
    # All the points above +T2
    positive_T2_crossing = (np.array(period_after_T1_crossing) > trigger_config['th2']).astype(int)
    # Positive crossing, the point before which is below T2.
    mask_T2_crossing_positive = np.diff(positive_T2_crossing) == 1
    # if np.sum(mask_T2_crossing_positive) > 0:
    #     index_T2_crossing_positive = np.arange(len(period_after_T1_crossing) - 1)[mask_T2_crossing_positive]
    negative_T2_crossing = (np.array(period_after_T1_crossing) < - trigger_config['th2']).astype(int)
    mask_T2_crossing_negative = np.diff(negative_T2_crossing) == 1
    # if np.sum(mask_T2_crossing_negative) > 0:
    #     index_T2_crossing_negative = np.arange(len(period_after_T1_crossing) - 1)[mask_T2_crossing_negative]
    # n_T2_crossing_negative = np.len(index_T2_crossing_positive)
    # Register the first T1 crossing as a T2 crossing
    mask_first_T1_crossing = np.zeros(len(period_after_T1_crossing), dtype=bool)
    mask_first_T1_crossing[0] = True
    mask_first_T1_crossing[1:] = (mask_T2_crossing_positive | mask_T2_crossing_negative)
    index_T2_crossing = np.arange(len(period_after_T1_crossing))[mask_first_T1_crossing]
    n_T2_crossing = 1 # Starting from the first T1 crossing.
    dict_trigger_infos["index_T2_crossing"] = [0]
    if len(index_T2_crossing) > 1:
      for i, j in zip(index_T2_crossing[:-1], index_T2_crossing[1:]):
          # The separation between successive T2 crossings
          time_separation = (j - i) * 2
          if time_separation <= trigger_config["t_sepmax"]:
              n_T2_crossing += 1
              dict_trigger_infos["index_T2_crossing"].append(j)
          else:
              # Violate the maximum separation, stop counting NC
              # Save the position of the last T2 crossing, i.e., i
              # to be used for calculating the Q value
            break
    else:
      n_T2_crossing = 1
      j = 1
    # Change the reference of indices of T2 crossing
    dict_trigger_infos["index_T2_crossing"] = np.array(dict_trigger_infos["index_T2_crossing"]) + dict_trigger_infos["index_T1_crossing"]
    dict_trigger_infos["NC"] = n_T2_crossing
    # Calulate the peak value
    dict_trigger_infos["Q"] = (np.max(np.abs(period_after_T1_crossing[:j])) - baseline) / dict_trigger_infos["NC"]
    return dict_trigger_infos


def search_windows(trace,
                   filter_status='off',
                   threshold1=55,
                   threshold2=35,
                   maxRMS=21,
                   num_cross=2+4,
                   num_separation=750,
                   num_interval=26,
                   sample_frequency=500,
                   cutoff_frequency=50,
                   samples_from_trace_edge=100):
    '''
    Adapted from Xishui Tian's transient search.
    Searches for pulse windows in a trace with a double-threshold trigger.
    Default values are determined from analysis with
    `test_pretrigger.py` and `find_pretrigger_threshold.ipynb`
    '''

    # samples_from_trace_edge = max(num_interval,samples_from_trace_edge)
    trace = trace[samples_from_trace_edge:trace.shape[-1]-samples_from_trace_edge]

    window_list     = []
    trigger_id_list = []

    # if filter_status == 'on':
    #     trace = signal.high_pass_filter(trace=trace, sample_frequency=sample_frequency, cutoff_frequency=cutoff_frequency)
 
    # stop if there are no transients/pulses
    cross_threshold1 = np.abs(trace) > threshold1
    cross_threshold2 = np.abs(trace) > threshold2
    # print(f'\n\n\n thresholds {threshold1, threshold2, np.sum(cross_threshold1), np.sum(cross_threshold2), num_cross}')
    if np.sum(cross_threshold1[num_interval:]) < 2 or np.sum(cross_threshold2[num_interval:]) < num_cross:
        #logger.debug(f'Not enough crossings: T1 = {np.sum(cross_threshold1[num_interval:])}; T2 {np.sum(cross_threshold2[num_interval:])}')
        return window_list, trigger_id_list

    # exclude abnormal traces
    # if get1trace_RMS(trace=trace) > maxRMS or max(get1trace_PSD(trace=trace)) > maxPSD:
    #     return []
    
    # find trace positions where threshold1 is exceeded
    cross1_ids = np.flatnonzero(cross_threshold1)
    cross1_ids = cross1_ids[cross1_ids>=num_interval] # first T1 crossing can only be after quiet time condition
    #logger.debug(cross1_ids)
    # find the separations between consecutive threshold crossings
    cross1_separations = np.diff(cross1_ids)

    # locate pulse indices in threshold crossing indices
    pulse_ids = np.flatnonzero(cross1_separations > num_separation)
    pulse_ids = np.concatenate(([-1], pulse_ids, [len(cross1_ids)-1]))
    
    # search all transients/pulses
    for i in range(len(pulse_ids)-1):
        trigger_id = cross1_ids[pulse_ids[i]+1]

        # quiet time condition before trigger
        if np.sum(cross_threshold1[trigger_id-num_interval:trigger_id]) > 0:
            #logger.debug(f'Pulse {i}: Quiet time condition not fulfilled: {np.sum(cross_threshold1[trigger_id-num_interval:trigger_id])}')
            continue

        # get the start index of current pulse
        start_id = trigger_id - num_interval
        start_id = max(0, start_id) # fix the 1st pulse

        # get the stop index of current pulse
        cross2_ids       = np.flatnonzero(cross_threshold2)[np.flatnonzero(cross_threshold2) >= trigger_id]
        cross2_intervals = np.diff(cross2_ids)
        pulse2_ids       = np.flatnonzero(cross2_intervals > num_interval)
        if len(pulse2_ids) == 0:
            stop_id = cross2_ids[-1] + num_interval
        else:
            stop_id = cross2_ids[pulse2_ids[0]] + num_interval
        stop_id = min(len(trace)-1, stop_id) # fix the last pulse
        
        if np.sum(cross_threshold1[start_id:stop_id+1]) >= 2 and np.sum(cross_threshold2[start_id:stop_id+1]) >= num_cross:
            #if np.sum(cross_threshold2[start_id:stop_id+1]) < 32: # This will not work for very high SNR signals compared to the thresholds
                #logger.debug('Too many T2 crossings')
            window_list.append([start_id+samples_from_trace_edge, stop_id+samples_from_trace_edge])
            trigger_id_list.append(trigger_id+samples_from_trace_edge)
            # else:
            #     print(np.sum(cross_threshold2[start_id:stop_id+1]))

    return window_list, trigger_id_list


def search_pulse(trace,
                 filter_status='on',
                 threshold1=60,
                 threshold2=36,
                 num_cross=2+4,
                 num_interval=16,
                 sample_frequency=500,
                 cutoff_frequency=50,
                 samples_from_trace_edge=0):
    '''
    Adapted from Xishui Tian's transient search.
    Searches for the FIRST pulse window in a trace with a double-threshold trigger.
    This mimics the trigger in hardware.
    MOD: added a parameter to avoid triggeres near the trace edge.
    '''

    trace = trace[samples_from_trace_edge:trace.shape[-1]-samples_from_trace_edge]

    # if filter_status == 'on':
    #     trace = high_pass_filter(trace=trace, sample_frequency=sample_frequency, cutoff_frequency=cutoff_frequency)

    # stop if there are no transients/pulses                                                                                                                                                                                                         
    cross_threshold1 = np.abs(trace) > threshold1
    cross_threshold2 = np.abs(trace) > threshold2
    if np.sum(cross_threshold1) < 2 or np.sum(cross_threshold2) < num_cross:
        logger.debug('Not enough T1 or T2 crossings')
        return [], []

    # exclude abnormal traces                                                                                                                                                                                                                        
    # if get1trace_RMS(trace=trace) > maxRMS or max(get1trace_PSD(trace=trace)) > maxPSD:
    #     return []

    # find trace positions where threshold1 is exceeded                                                                                                                                                                                              
    cross1_ids = np.flatnonzero(cross_threshold1)

    # find the separations between consecutive threshold crossings                                                                                                                                                                                   
    cross1_separations = np.diff(cross1_ids)

    # locate pulse indices in threshold crossing indices                                                                                                                                                                                             
    pulse_ids = np.flatnonzero(cross1_separations > num_interval)
    pulse_ids = np.concatenate(([-1], pulse_ids, [len(cross1_ids)-1]))

    # search all transients/pulses                                                                                                                                                                                                                   
    for i in range(len(pulse_ids)-1):
        trigger_id = cross1_ids[pulse_ids[i]+1]
        # if np.sum(cross_threshold2[trigger_id-num_interval:trigger_id]) > 2:
        #     continue
        
        if np.sum(cross_threshold1[trigger_id-num_interval:trigger_id]) > 0:
            logger.debug('Quiet time condition not fulfilled')
            #print('test',np.sum(cross_threshold1[trigger_id-num_interval:trigger_id]))
            continue

        # get the start index of current pulse                                                                                                                                                                                                       
        start_id = trigger_id - num_interval
        start_id = max(0, start_id) # fix the 1st pulse                                                                                                                                                                                              

        # get the stop index of current pulse                                                                                                                                                                                                        
        cross2_ids       = np.flatnonzero(cross_threshold2)[np.flatnonzero(cross_threshold2) >= trigger_id]
        cross2_intervals = np.diff(cross2_ids)
        pulse2_ids       = np.flatnonzero(cross2_intervals > num_interval)
        if len(pulse2_ids) == 0:
            stop_id = cross2_ids[-1] + num_interval
        else:
            stop_id = cross2_ids[pulse2_ids[0]] + num_interval
        stop_id = min(len(trace)-1, stop_id) # fix the last pulse                                                                                                                                                                                    
        #print([start_id+samples_from_trace_edge,stop_id+samples_from_trace_edge])
        #print(np.sum(cross_threshold1[start_id:stop_id+1]),np.sum(cross_threshold2[start_id:stop_id+1]))
        
        if np.sum(cross_threshold1[start_id:stop_id+1]) >= 2 and np.sum(cross_threshold2[start_id:stop_id+1]) >= num_cross:
            #if np.sum(cross_threshold2[start_id:stop_id+1]) < 32: # This will not work for very high SNR signals compared to the thresholds
            #if np.sum(cross_threshold2[start_id:stop_id+1]) - np.sum(cross_threshold1[start_id:stop_id+1]) < 32-2:
            return [[start_id+samples_from_trace_edge, stop_id+samples_from_trace_edge]], [trigger_id+samples_from_trace_edge]

    return [], []


def thresh_trigger(trace,
                   include_Z=False,
                   **kwargs):
    '''
    Adapted from Xishui Tian's transient search.
    Performs the double-threshold trigger for a given trace.
    '''

    channels = ['X','Y','Z']
    if not include_Z:
        channels = channels[:2]
    
    # Search for pulses                                                                                                                                                                                                                                                         
    num_pulse = 0
    list_trigger_flag = [0, 0, 0]
    trigger_pos_ch = -1*np.ones(3,dtype=int) # the 'trigger position' is the sample where the first pulse is recorded

    for i, channel in enumerate(channels):                                                                                                                                                                                                  
        windows, trigger_ids = search_windows(trace[i], **kwargs) #trace=high_pass_filter(trace[i])
        num_pulse += len(windows)
        if len(windows):
            list_trigger_flag[i] = 1
            trigger_pos_ch[i] = trigger_ids[0]

    # The final trigger position is where is where the first trigger occurs in all channels
    if num_pulse:
        trigger_pos = np.min(trigger_pos_ch[trigger_pos_ch>=0])
    else:
        trigger_pos = -1

    if list_trigger_flag == [0, 0, 0]:
        # No trigger                                                                                                                                                                                                                                                            
        trigger_flag = 0
    if list_trigger_flag == [1, 0, 0]:
        # X trigger                                                                                                                                                                                                                                                             
        trigger_flag = 1
    if list_trigger_flag == [0, 1, 0]:
        # Y trigger                                                                                                                                                                                                                                                             
        trigger_flag = 2
    if list_trigger_flag == [0, 0, 1]:
        # Z trigger                                                                                                                                                                                                                                                             
        trigger_flag = 3
    if list_trigger_flag == [1, 1, 0]:
        # X Y trigger                                                                                                                                                                                                                                                           
        trigger_flag = 12
    if list_trigger_flag == [1, 0, 1]:
        # X Z trigger                                                                                                                                                                                                                                                           
        trigger_flag = 13
    if list_trigger_flag == [0, 1, 1]:
        # Y Z trigger                                                                                                                                                                                                                                                           
        trigger_flag = 23
    if list_trigger_flag == [1, 1, 1]:
        # X Y Z trigger                                                                                                                                                                                                                                                           
        trigger_flag = 123

    return num_pulse, trigger_flag, trigger_pos


def find_thresh_triggers(traces,
                         **kwargs):

    trigger_flags = np.zeros(traces.shape[0],dtype=int)
    trigger_times = -1*np.ones(traces.shape[0],dtype=int)

    for i, trace in enumerate(traces):
        _, trigger_flag, trigger_time = thresh_trigger(trace,**kwargs)
        
        trigger_flags[i] = trigger_flag
        trigger_times[i] = trigger_time

    return trigger_flags, trigger_times


def filter_traces(traces,
                  freq_highpass=50.,
                  freqs_notch=[50.2,55.1,126],
                  bw_notch=[1.,1.,25.],
                  sampling_rate=500.):
    '''
    Bandpass filter above > 43 MHz to kill short waves
    Notch filters to kill communication lines

    Default values are determined from analysis with
    `test_pretrigger.py` and `find_pretrigger_threshold.ipynb`
    '''

    if type(bw_notch) == float:
        bw_notch = np.ones(len(freqs_notch))*bw_notch
    else:
        assert len(bw_notch) == len(freqs_notch)

    window = signal.windows.general_gaussian(traces.shape[-1],10,traces.shape[-1]/2.3)
    traces = window*traces

    sos             = signal.butter(10,freq_highpass,btype='high',analog=False,fs=sampling_rate,output='sos')
    traces_filtered = signal.sosfiltfilt(sos,traces)

    for freq, bw in zip(freqs_notch,bw_notch):
        # b, a            = signal.iirnotch(freq,freq/bw,fs=sampling_rate)
        # traces_filtered = signal.filtfilt(b,a,traces_filtered)
        freq_bandstop   = [freq-bw/2.,freq+bw/2.]
        sos             = signal.butter(10,freq_bandstop,btype='bandstop',analog=False,fs=sampling_rate,output='sos')
        traces_filtered = signal.sosfiltfilt(sos,traces_filtered)

    return np.trunc(traces_filtered)


def get_masks_du(du_ids):

    masks_du = {du : np.zeros(du_ids.shape,dtype=bool) for du in np.unique(du_ids)}

    for du in masks_du.keys():
        for i, du_id in enumerate(du_ids):
            if du == du_id:
                masks_du[du][i] = True
                
    return masks_du


def get_pulse_inj_weights(bkg_pulse_file):

    assert os.path.splitext(bkg_pulse_file)[1] == '.npz', f'Pulse file does not have the correct extension {os.path.splitext(bkg_pulse_file)[1]}'

    logger.info(f'Getting injection time weights from {bkg_pulse_file}')

    f             = np.load(bkg_pulse_file)
    n_samples     = f['traces'].shape[-1]
    pretrig_times = f['pretrig_times']

    bin_edges  = np.arange(n_samples+1)
    weights, _ = np.histogram(pretrig_times,bins=bin_edges,density=True)

    return weights


def get_noise_traces(noise_dir,
                     n_traces,
                     rng=np.random.default_rng()):
    
    noise_files = sorted( glob.glob( os.path.join(noise_dir,'*.npz') ) ) # sort to get rid of glob randomness
    noise_file  = rng.choice(noise_files)

    logger.info(f'Selecting {n_traces} random noise traces from {noise_file}')

    with np.load(noise_file) as f:
        entries      = rng.choice( np.arange( f['traces'].shape[0] ),size=n_traces,replace=False )
        noise_traces = f['traces'][entries]

    return noise_traces, noise_file, entries


def add_sim_to_noise(sim_traces,
                     noise_traces,
                     inj_weights=None,
                     rng=np.random.default_rng(),
                     adc_saturation=2**13,
                     offset=15):
    
    logger.info(f'Adding {sim_traces.shape[0]} simulated pulses to noise traces at random trace positions')

    sim_traces = set_trace_length(sim_traces,noise_traces.shape[-1])
    n_traces   = sim_traces.shape[0]
    n_samples  = sim_traces.shape[-1]

    #-#-#- Get random injection times according to given weights  -#-#-#
    if type(inj_weights) == type(None):
        logger.warning('No weights given. Pulses are added uniformly over entire trace.')
        inj_weights  = np.ones(n_samples)
        inj_weights /= inj_weights.sum()
    else:
        assert noise_traces.shape[-1] == inj_weights.shape[-1]

    max_sample_inj  = rng.choice(np.arange(n_samples),size=n_traces,p=inj_weights)
    
    #-#-#- Find the corresponding 'start time' of the pulse with peak at the injection time -#-#-#
    sim_trace_abs   = np.abs(sim_traces)
    max_samples_sim = np.argmax( sim_trace_abs,axis=2,keepdims=True ) 
    max_sim         = np.take_along_axis( sim_trace_abs,max_samples_sim,axis=2 )
    max_sim_pol     = np.argmax( max_sim[...,0],axis=1,keepdims=True )
    max_sample_sim  = np.take_along_axis( max_samples_sim[...,0],max_sim_pol,axis=1 )[...,0]

    pulse_start_inj = max_sample_inj - max_sample_sim + offset + n_samples


    # max_sample_rand  = rng.integers(100,n_samples-100,n_traces) # not near edges of trace

    #-#-#- Add simulated air-shower pulse to noise trace  -#-#-#
    sig_traces        = np.zeros(sim_traces.shape)
    noise_range_dummy = np.arange(n_samples,2*n_samples,dtype=int)

    for i in range(n_traces):
        sig_trace       = np.zeros((3,n_samples*3))
        sig_range_dummy = np.arange(pulse_start_inj[i],pulse_start_inj[i]+n_samples,dtype=int)

        sig_trace[:,noise_range_dummy] += noise_traces[i]        
        sig_trace[:,sig_range_dummy]   += sim_traces[i]

        sig_traces[i] = sig_trace[:,noise_range_dummy]

    #-#-#- Saturate a possible signal at the ADC limit -#-#-#
    sig_traces = np.where(np.abs(sig_traces)<adc_saturation,sig_traces,np.sign(sig_traces)*adc_saturation)

    return sig_traces, max_sample_inj


def rms(trace,
        samples_from_trace_edge=100,
        **kwargs):
    
    trace_without_edges = trace[...,samples_from_trace_edge:trace.shape[-1]-samples_from_trace_edge]
    
    rms = np.sqrt( np.mean( trace_without_edges**2,**kwargs ) )

    return rms


def get_snr(sig_traces,
            inj_pulse_times,
            jitter_size=20,
            window_size=100):
    
    snr = np.zeros( (sig_traces.shape[0],sig_traces.shape[1]) )

    for i, sig_trace in enumerate(sig_traces):
        inj_pulse_time = inj_pulse_times[i]
        max_pulse      = np.max( np.abs( sig_trace[...,inj_pulse_time-jitter_size:inj_pulse_time+jitter_size] ),axis=1 )

        window_pulse = np.arange( np.max( [0,inj_pulse_time-window_size] ),
                                  np.min( [sig_trace.shape[-1],inj_pulse_time+window_size] ) )
        
        trace_without_pulse = np.delete(sig_trace,window_pulse,axis=1)
        rms_without_pulse   = rms(trace_without_pulse,axis=1)

        snr[i] = max_pulse / rms_without_pulse
    
    return snr