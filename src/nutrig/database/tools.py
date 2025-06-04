###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import glob
import sys
import logging

import numpy as np
from scipy.signal import lfilter, minimum_phase
from scipy.signal.windows import gaussian

import grand.dataio.root_trees as rt
from grand.geo.coordinates import Geodetic, GRANDCS

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


def get_baseline_calib(baseline_dir='/sps/grand/pcorrea/gp80/baseline_calibration/RUN148'):
    '''
    Obtain the baseline calibration for each channel of each DU from Nathan's analysis of RUN 145.
    '''
    # One file of mean baselines per channel
    files = sorted( glob.glob( os.path.join(baseline_dir,'*.txt') ) )
    
    du_ids         = np.loadtxt(files[0],skiprows=0,usecols=0,delimiter=':').astype(int)
    baseline_calib = {du : np.zeros(3,dtype=int) for du in du_ids}

    for ch, file in enumerate(files):
        # Round to nearest integer since ADC counts are integer
        baselines_ch = np.round( np.loadtxt(file,skiprows=0,usecols=1) )
        for j, du in enumerate(du_ids):
            baseline_calib[du][ch] = baselines_ch[j]

    return baseline_calib


def get_gauss_window_edges(fade,
                           n_samples,
                           steepness=4):
    '''
    Obtain a symmetric window with a Gaussian fade-in and fade-out.

    Arguments
    ---------
    `fade`
    type        : int
    description : Number of samples over which to apply the fade.

    `n_samples`
    type        : int
    description : Total number of samples in a trace.

    `steepness`
    type        : float
    description : Steepness of the Gaussian edges.
    
                                
    Returns
    -------
    `window`
    type        : np.ndarray[float]
    description : The window, normalized between [0,1], with shape (N_samples).
    '''

    std = fade/steepness
    gauss = gaussian(fade*2,std)
    gauss /= np.max(gauss) # normalize to 1

    fade_in  = gauss[:fade]
    fade_out = gauss[-fade:]

    window = np.ones(n_samples)
    window[:fade] = fade_in
    window[-fade:] = fade_out

    return window


def apply_notch_filter(trace,
                       f_notch,
                       r,
                       f_sample=5e8):

    '''
    Notch filter implementation, mimicking the notch filter on the FEB.
    Adapted from Kato Sei's script.
    
    Arguments
    ---------
    `trace`
    type        : np.ndarray[int]
    units       : ADC counts
    description : Single ADC trace for one channel, with shape (N_samples).

    `f_notch`
    type        : float
    units       : Hz
    description : Frequency of the notch filter.

    `r`
    type        : float
    units       : Hz
    description : Radius of the notch filter.

    `f_sample`
    type        : float
    units       : Hz
    description : Sampling rate of the ADC. Currently 500 MHz.
    
                                
    Returns
    -------
    `y`
    type        : np.ndarray[float]
    units       : "ADC counts"
    description : The filtered trace, with shape (N_samples).
    '''
    
    nu = 2. * np.pi * f_notch / f_sample

    ### Calculation of coefficients
    a1 = 2. * (r ** 4) * np.cos(4.*nu)
    a2 = - (r ** 8)
    b1 = - 2. * np.cos(nu)
    b2 = 1
    b3 = 2. * r * np.cos(nu)
    b4 = r * r
    b5 = 2. * r * r * np.cos(2.*nu)
    b6 = r ** 4

    ### Calculation of the trace after passing the digital notch filter
    ### Parameters:
    ### y[n_sample]: output trace
    ### y1[n_sample] & y2[n_sample]: intermediate variables

    # Try reflection padding
    padding_idx = 0 # up to which index you want to do the padding, 0 does nothing

    trace_padded = np.concatenate((trace[:padding_idx][::-1],trace))
    y, y1, y2 = np.zeros(trace_padded.shape[0]), np.zeros(trace_padded.shape[0]), np.zeros(trace_padded.shape[0])

    #beginning at sample 10 to avoid the artificial peak
    #start, end = 0, trace.shape[1] #1024 for MD data and 2048 for DC2
    for n in range(trace_padded.shape[0]):
    #for n in range(start, end):
        y1[n] = b2 * trace_padded[n] + b1 * trace_padded[n-1] + trace_padded[n-2]
        y2[n] = y1[n] + b3 * y1[n-1] + b4 * y1[n-2]
        y[n]  = a1 * y[n-4] + a2 * y[n-8] + y2[n-2] + b5 * y2[n-4] + b6 * y2[n-6]

    return y[padding_idx:]


def filter_traces_notch(traces):
    '''
    Wrapper for the `apply_notch_filter` function.
    Performs all notch filters for an array of multiple traces.

    Arguments
    ---------
    - `traces`
      + type        : `np.ndarray[int]`
      + units       : ADC counts
      + description : Array of ADC traces, with shape `(N_traces,N_channels,N_samples)`.
                                
    Returns
    -------
    - `traces_filtered`
      + type        : `np.ndarray[float]`
      + units       : "ADC counts"
      + description : Array of filtered ADC traces, with shape `(N_traces,N_channels,N_samples)`.
    '''

    freqs_notch = np.array( [39, 119.4, 132, 137.8, 121.5, 134.2, 124.7, 119.2] )*1e6 # [Hz]
    radii_notch = np.array( [0.9, 0.98, 0.95, 0.98, 0.96, 0.96, 0.96, 0.98] )
    
    traces_filtered = np.zeros( traces.shape,dtype=int )
    
    logger.info(f'Applying notch filters...')
    for f_notch, r in zip(freqs_notch,radii_notch):
        logger.info(f'at frequency {f_notch/1e6:.2f} MHz with radius {r:.2f}')

    for i, trace in enumerate(traces):
        for ch in range( traces.shape[1] ):
            trace_filtered = trace[ch].copy()
            for f_notch, r in zip(freqs_notch,radii_notch):
                trace_filtered = apply_notch_filter(trace_filtered,f_notch,r)
            traces_filtered[i,ch] = trace_filtered

    logger.info(f'Traces filtered!')

    return traces_filtered


def filter_traces_bandpass(traces,
                           coeff_file='/sps/grand/pcorrea/nutrig/database/v2/lowpass115MHz.txt',
                           do_minimum_phase=False):
    '''
    Mimics the DIRECT form FIR band-pass filter < 115 MHz that is implemented on the online firmware.
    Filter coefficients are provided by Xing Xu.
    Implemented with help of ChatGPT. See also `./test_bandpass_filter.ipynb`.

    Arguments
    ---------
    - `traces`
        + type        : `np.ndarray[int]`
        + units       : ADC counts
        + description : Array of ADC traces, with shape `(N_traces,N_channels,N_samples)`.

    - `coeff_file`
        + type        : str
        + description : File containing the filter coefficients.

    - `do_minimum_phase`
        + type        : bool
        + description : Option to only keep minimum phase coefficients for the filter.
                                
    Returns
    -------
    - `traces_filtered`
        + type        : `np.ndarray[float]`
        + units       : "ADC counts"
        + description : Array of filtered ADC traces, with shape `(N_traces,N_channels,N_samples)`.
    '''

    logger.info(f'Applying band-pass filter with cutoff frequency at 115 MHz...')

    coeff = np.loadtxt(coeff_file,delimiter=',')

    # This will get rid of non-physical ripples before simulated pulses
    # It will also get rid of the phase delay; normally this is (N-1)/2 for N coefficients
    if do_minimum_phase:
        coeff = minimum_phase(coeff,method='hilbert')

    traces_filtered = lfilter(coeff,1,traces)

    return traces_filtered.astype(traces.dtype)


def trigger_FLT0(trace,
                 dict_trigger_parameters,
                 samples_from_edge=100):
    '''
    Evaluates the FLT-0 trigger, mimicking the L1 trigger on the FEB.
    Adapted from Marion Guelfand's script.

    Arguments
    ---------
    - `trace`
      + type        : `np.ndarray[int]`
      + units       : ADC counts
      + description : A single ADC trace, with shape `(N_samples)`.

    - `dict_trigger_parameters`
      + type        : `dict`
      + fields
        * `th1` [`int`] : T1 or signal threshold [ADC counts].
        * `th2` [`int`] : T2 or noise threshold [ADC counts].
        * `t_quiet` [`int`] : Quiet time before T1 threshold [ns].
        * `t_period` [`int`] : Time period over which the number of T2 crossings are evaluated [ns].
        * `t_sepmax` [`int`] : Maximum time allowed between two T2 crossings [ns].
        * `nc_min` [`int`] : Minimum number of T2 crossings.
        * `nc_max` [`int`] : Maximum number of T2 crossings.

    - `samples_from_edge`
      + type        : `int`
      + units       : ADC samples of 2 ns
      + description : Number of samples to ignore from the edges of the trace.
                      Typically required to mitigate boundary effects caused by filtering.
                      For the notch filter of Kato Sei it seems to only be an issue at the start of the trace.
                                
    Returns
    -------
    - `traces_filtered`
      + type        : `np.ndarray[float]`
      + units       : "ADC counts"
      + description : Array of filtered ADC traces, with shape `(N_traces,N_du,N_samples)`.
    
    '''
    # Find the indices where the signal crosses the first threshold (T1)
    index_t1_crossing = np.where(trace > dict_trigger_parameters["th1"])[0]
    
    # Lists to store results
    T1_indices    = []  # Indices of T1 crossings
    T1_amplitudes = []  # Amplitudes at T1 crossings
    NC_values     = []  # Number of T2 crossings for each T1 (the T1 is included in NC)

    # Compute the first sample from which T_quiet can be evaluated given boundary conditions
    sample_min = samples_from_edge #+ dict_trigger_parameters['t_quiet']//2

    # Compute the last sample up to which T_period can be evaluated given boundary conditions
    sample_max = len(trace) - dict_trigger_parameters['t_period']//2 #- samples_from_edge

    # Process each T1 crossing
    for index_T1 in index_t1_crossing:
        # Check if the T1 index is greater than 25 (bug linked to the notch filter: artificial peak that appears)
        ## to be corrected
        if index_T1 <= sample_min:
            logger.debug(f'Skipping T1 @ index = {index_T1}: too close to edge')
            continue  # Skip this T1 if its index is not greater than 25 (100 for sims ZHAireS)

        if index_T1 >= sample_max:
            logger.debug(f'Stopping: T_period cannot be evaluated for T1 @ index >= {index_T1}')
            break

        logger.debug(f'Now checking T1 @ index = {index_T1}')
        
        start = max(samples_from_edge, index_T1 - dict_trigger_parameters['t_quiet'] // 2)
        end = index_T1
        
        # Check if the signal before T1 is below the first threshold (T1 is valid)
        if np.all(trace[start:end] <= dict_trigger_parameters["th1"]):
            # Extract the period after T1 to find T2 crossings
            period_after_T1 = trace[index_T1: index_T1 + dict_trigger_parameters['t_period'] // 2]
            positive_T2_crossing = (period_after_T1 > dict_trigger_parameters['th2']).astype(int)
            
            # Search for positive T2 crossings
            mask_T2_crossing_positive = np.diff(positive_T2_crossing) == 1
            index_T2 = np.where(mask_T2_crossing_positive)[0] + 1 + index_T1
            index_T2 = np.insert(index_T2, 0, index_T1) #add T1 in index crossings
            n_T2_crossing = len(index_T2)   # Number of T2 crossings (including the T1 crossing itself)
            valid_T1 = True

            # Check for maximum separation condition between T2 crossings
            for i, j in zip(index_T2[:-1], index_T2[1:]):
                time_separation = (j - i) * 2  # Calculate the time separation between consecutive T2 crossings
                
                # If the separation is too large, mark T1 as invalid
                if time_separation > dict_trigger_parameters["t_sepmax"]:
                    valid_T1 = False
                    logger.debug(f'Killed by Tsepmax < {time_separation} ns')
                    break

            # If the number of T2 crossings is out of bounds, ignore this T1
            # NOTE: online we exclude also when NC == NC_max (NC_min)
            if n_T2_crossing <= dict_trigger_parameters["nc_min"] or n_T2_crossing >= dict_trigger_parameters["nc_max"]:
                valid_T1 = False
                logger.debug(f'Killed by NC = {n_T2_crossing} >= NC_max (or <= NC_min)')

            if valid_T1:
                # If T1 is valid, record the number of T2 crossings
                NC_values.append(n_T2_crossing)
                T1_indices.append(index_T1)
                T1_amplitudes.append(trace[index_T1])
                logger.debug(f'OK! N_crossings = {n_T2_crossing}')

        else:
            logger.debug(f'Killed by {len( np.where( trace[start:end] > dict_trigger_parameters["th1"] )[0] )} T1 crossings within T_quiet')

      
    if T1_indices:
        return T1_indices, T1_amplitudes, NC_values
    else:
        return  # Do nothing if no valid T1 crossings were found
    

def do_FLT0(traces,
            *args,
            channels=[0,1],
            **kwargs):
    '''
    Wrapper for the `trigger_FLT0` function.
    Performs the FLT-0 for an array of multiple traces.
    Only the first FLT-0 trigger is recorded if there are multiple pulses.
    With default parameters it is rare to have multiple FLT-0 triggers in one trace.

    Arguments
    ---------
    - `traces`
      + type        : `np.ndarray[int]`
      + units       : ADC counts
      + description : Array of ADC traces, with shape `(N_traces,N_channels,N_samples)`.

    - `*args`
      + description : Positional arguments to pass to `trigger_FLT0`.

    - `channels`
      + type        : `list[int]`
      + description : Channels for which to evaluate the FLT-0.
                      Default assumes that trace has three channels (0,1,2) = (X,Y,Z) and only evaluates X and Y.

    - `**kwargs`
      + description : Keyword arguments to pass to `trigger_FLT0`.
                                
    Returns
    -------
    - `FLT0_flags`
      + type        : `np.ndarray[bool]`
      + description : Flags of the FLT-0. `True` for a trigger, `False` if there was no trigger. Shape: `(N_traces,N_channels)`.

    - `FLT0_first_T1_idx`
      + type        : `np.ndarray[int]`
      + description : Sample/index of the first T1 crossing corresponding to an FLT-0. `-1` if there was no trigger. Shape: `(N_traces,N_channels)`.

    - `n_FLT0`
      + type        : `np.ndarray[int]`
      + description : Number of FLT-0 triggers for each trace. Shape: `(N_traces,N_channels)`.
    '''

    logger.info(f'Performing FLT-0 trigger for {len(traces)} traces...')

    FLT0_flags         = np.zeros( (traces.shape[0],traces.shape[1]),dtype=bool )
    FLT0_first_T1_idcs = -1 * np.ones( (traces.shape[0],traces.shape[1]),dtype=int )
    n_FLT0             = np.zeros( (traces.shape[0],traces.shape[1]),dtype=int )
    
    for i, trace in enumerate(traces):
        logger.debug(f'Searching triggers in trace {i}')
        for ch in channels:
            logger.debug(f'Channel {ch}')

            try:
                T1_indices, T1_amplitudes, NC_values = trigger_FLT0(trace[ch],*args,**kwargs)
                FLT0_flags[i,ch]                     = True
                FLT0_first_T1_idcs[i,ch]             = T1_indices[0]
                n_FLT0[i,ch]                        += len(T1_indices)

            except:
                continue

    res_FLT0 = {'FLT0_flags' : FLT0_flags,
                'FLT0_first_T1_idcs' : FLT0_first_T1_idcs,
                'n_FLT0' : n_FLT0}

    return res_FLT0


def compute_FLT0_trigger_rate_MD(n_FLT0,
                                 FLT0_flags,
                                 du_ids,
                                 du_seconds,
                                 du_nanoseconds,
                                 t_eff,
                                 channels=[0,1]):

    # Get unique DU IDs and the corresponding number of entries
    du_ids_unique, n_entries_per_du = np.unique(du_ids,return_counts=True)

    t_bin               = np.zeros( du_ids_unique.shape[0] ) # time bin over which the rate is computed
    trigger_rate_per_ch = np.zeros( ( du_ids_unique.shape[0],n_FLT0.shape[1] ) )
    trigger_rate_OR     = np.zeros( du_ids_unique.shape[0] )
    trigger_rate_AND    = np.zeros( du_ids_unique.shape[0] )

    mask_OR  = np.any( FLT0_flags[:,channels],axis=1 )
    mask_AND = np.all( FLT0_flags[:,channels],axis=1 )

    for i, du_id in enumerate(du_ids_unique):
        t_du     = du_seconds[du_ids==du_id] + du_nanoseconds[du_ids==du_id]*1e-9 # [s]
        t_bin[i] = t_du[-1] - t_du[0] # [s]

        n_FLT0_du    = n_FLT0[du_ids==du_id]
        t_eff_tot_du = n_entries_per_du[i] * t_eff # [s]

        mask_OR_du  = mask_OR[du_ids==du_id]
        mask_AND_du = mask_AND[du_ids==du_id]

        trigger_rate_per_ch[i] = np.sum( n_FLT0_du,axis=0 ) / t_eff_tot_du # [Hz]
        trigger_rate_OR[i]     = np.sum( np.max( n_FLT0_du[mask_OR_du],axis=1 ) ) / t_eff_tot_du # [Hz]
        trigger_rate_AND[i]    = np.sum( np.max( n_FLT0_du[mask_AND_du],axis=1 ) ) / t_eff_tot_du # [Hz]

    res_FLT0_trigger_rate = {'trigger_rate_per_ch' : trigger_rate_per_ch,
                             'trigger_rate_OR' : trigger_rate_OR,
                             'trigger_rate_AND' : trigger_rate_AND,
                             't_bin' : t_bin}

    return res_FLT0_trigger_rate


def compute_FLT0_trigger_rate_UD(n_FLT0,
                                 FLT0_flags,
                                 du_ids,
                                 du_seconds,
                                 du_nanoseconds,
                                 channels=[0,1]):

    # Get unique DU IDs and the corresponding number of entries
    du_ids_unique, n_entries_per_du = np.unique(du_ids,return_counts=True)

    t_bin               = np.zeros( du_ids_unique.shape[0] ) # time bin over which the rate is computed
    trigger_rate_per_ch = np.zeros( ( du_ids_unique.shape[0],n_FLT0.shape[1] ) )
    trigger_rate_OR     = np.zeros( du_ids_unique.shape[0] )
    trigger_rate_AND    = np.zeros( du_ids_unique.shape[0] )

    mask_OR  = np.any( FLT0_flags[:,channels],axis=1 )
    mask_AND = np.all( FLT0_flags[:,channels],axis=1 )

    for i, du_id in enumerate(du_ids_unique):
        t_du     = np.sort( du_seconds[du_ids==du_id] + du_nanoseconds[du_ids==du_id]*1e-9 ) # [s]
        t_bin[i] = t_du[-1] - t_du[0] # [s]

        n_FLT0_du = n_FLT0[du_ids==du_id]

        mask_OR_du  = mask_OR[du_ids==du_id]
        mask_AND_du = mask_AND[du_ids==du_id]

        trigger_rate_per_ch[i] = np.sum( n_FLT0_du,axis=0 ) / t_bin[i] # [Hz]
        trigger_rate_OR[i]     = np.sum( np.max( n_FLT0_du[mask_OR_du],axis=1 ) ) / t_bin[i] # [Hz]
        trigger_rate_AND[i]    = np.sum( np.max( n_FLT0_du[mask_AND_du],axis=1 ) ) / t_bin[i] # [Hz]

    res_FLT0_trigger_rate = {'trigger_rate_per_ch' : trigger_rate_per_ch,
                             'trigger_rate_OR' : trigger_rate_OR,
                             'trigger_rate_AND' : trigger_rate_AND,
                             't_bin' : t_bin}

    return res_FLT0_trigger_rate


def get_snr_and_t_pulse(traces,
                        FLT0_flags,
                        FLT0_first_T1_idcs,
                        samples_from_edge=100,
                        window_size_max=100):
    '''
    Computes the SNR and pulse-peak times for pulses in a data sample.
    The SNR per channel is defined as max(trace_ch)/rms(trace_ch).
    The SNR of the event, outputted here, is the maximum of the SNRs of the channels over which there was an FLT-0.
    The maximum is computed in a window of size `max_range` samples after the first T1 threshold index.
    The RMS is computed for all the samples after the above window.
    Samples close to the trace edge are ignored to mitigate boundary effects from filtering.

    Arguments
    ---------
    - `traces`
        + type        : `np.ndarray[int]`
        + units       : ADC counts
        + description : Array of ADC traces, with shape `(N_traces,N_channels,N_samples)`.

    - `FLT0_flags`
        + type        : `np.ndarray[bool]`
        + description : Flags of the FLT-0. `True` for a trigger, `False` if there was no trigger. Shape: `(N_traces,N_channels)`.

    - `FLT0_first_T1_idcs`
        + type        : `np.ndarray[int]`
        + description : Sample/index of the first T1 crossing corresponding to an FLT-0. `-1` if there was no trigger. Shape: `(N_traces,N_channels)`.

    - `samples_from_edge`
        + type        : `int`
        + units       : ADC samples of 2 ns
        + description : Number of samples to ignore from the edges of the trace.
                        Typically required to mitigate boundary effects caused by filtering.

    - `window_size_max`
        + type        : `int`
        + units       : ADC samples of 2 ns
        + description : Number of samples to consider, starting from the first T1 threshold crossing, for the computation of the trace maximum.
                                
    Returns
    -------
    - `snr`
        + type        : `np.ndarray[float]`
        + description : SNR for all input traces. 0 if the trace does not contain a pulse. Shape: `(N_traces)`.

    - `snr`
        + type        : `np.ndarray[float]`
        + description : Pulse-peak time for all input traces. -1 if the trace does not contain a pulse. Shape: `(N_traces,N_channels)`.
    '''
    
    snr_ch  = np.zeros( FLT0_flags.shape,dtype=float )
    t_pulse = np.ones( FLT0_flags.shape,dtype=int )
    n_entries, n_channels, n_samples = traces.shape

    for i in range(n_entries):
        for ch in range(n_channels):
            if FLT0_flags[i,ch]:
                # Define the samples over which to compute the maximum and RMS
                mask_max = np.arange( FLT0_first_T1_idcs[i,ch], FLT0_first_T1_idcs[i,ch]+window_size_max )
                mask_rms = np.arange( FLT0_first_T1_idcs[i,ch]+window_size_max, n_samples-samples_from_edge )

                max_ch = np.max( traces[i,ch,mask_max] )
                rms_ch = np.sqrt( np.mean( traces[i,ch,mask_rms]**2 ) )

                snr_ch[i,ch]  = max_ch/rms_ch
                t_pulse[i,ch] = FLT0_first_T1_idcs[i,ch] + np.argmax( traces[i,ch,mask_max] )
    
    # The final SNR of an event is the maximum of all the SNRs per channel
    snr = np.max(snr_ch,axis=1)

    return snr, t_pulse


def select_pulses_per_snr_bin(data,
                              snr_bins=np.linspace(3,8,6),
                              target_entries_per_bin=1000,
                              mode_snr='UNIFORM',
                              n_channels=3,
                              n_samples=1024):
    
    logger.info(f'Selecting traces in the following SNR bins: {snr_bins}')
    logger.info(f'Selection mode: {mode_snr}')
    
    selected_data = {}

    selected_data['traces']     = np.zeros( (0,n_channels,n_samples),dtype=int )
    selected_data['FLT0_flags'] = np.zeros( (0,n_channels), dtype=bool )
    selected_data['snr']        = np.zeros( (0,),dtype=float )
    selected_data['t_pulse']    = np.zeros( (0,n_channels),dtype=int )
    
    if mode_snr == 'UNIFORM':
        logger.info(f'Targeting {target_entries_per_bin} entries per bin...')

        for i in range( len( snr_bins ) - 1 ):
            idcs_snr_bin = np.where( np.logical_and( data['snr'] >= snr_bins[i], data['snr'] < snr_bins[i+1] ) )[0]

            size_sel = min( target_entries_per_bin,len(idcs_snr_bin) )
            idcs_sel = np.random.choice(idcs_snr_bin, size=size_sel, replace=False)
            
            selected_data['traces']     = np.vstack( (selected_data['traces'],data['traces'][idcs_sel]) )
            selected_data['FLT0_flags'] = np.vstack( (selected_data['FLT0_flags'],data['FLT0_flags'][idcs_sel]) )
            selected_data['snr']        = np.hstack( (selected_data['snr'],data['snr'][idcs_sel]) )
            selected_data['t_pulse']    = np.vstack( (selected_data['t_pulse'],data['t_pulse'][idcs_sel]) )

            logger.info(f'Selected {size_sel} background pulses in SNR bin [{snr_bins[i]},{snr_bins[i+1]}]')

    return selected_data


def get_du_xyz(
    long,
    lat,
    alt,
    obstime='2024-01-01', # other times, like the TRUE observation time, don't seem to work...
    origin=Geodetic(
        latitude=40.99746387,
        longitude=93.94868871,
        height=0) # lat, lon of the center station (from FEB@rocket) # z=0 @ sea level)
):
    '''
    Adapted from Marion Guelfand's script.
    '''

    logger.debug(origin)
    logger.debug(obstime)

    logger.debug('hey')
    # From GPS to Cartisian coordinates
    geod = Geodetic(latitude=lat, longitude=long, height=alt)
    logger.debug(geod.T.shape)
    gcs  = GRANDCS(geod, obstime=obstime, location=origin)

    

    logger.debug(gcs.T.shape)

    return gcs.T