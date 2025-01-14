#! /usr/bin/env python3
'''
DESCRIPTION
'''


###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import glob
import os
import argparse

import numpy as np
from scipy import signal

from nutrig.simu_analysis.tools import *


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def load_sig_traces_with_noise(sig_dir,
                               trace_length=1024):
    
    print(f'Loading signal+noise traces from {sig_dir}...')

    sig_files = sorted( glob.glob(sig_dir+'/sig_traces_with_noise_*.npz') )
    traces    = np.zeros( (1,3,trace_length) )
    snr       = np.zeros( (1,3) )

    for sig_file in sig_files[:]:
        f      = np.load(sig_file)
        traces = np.concatenate( (traces,f['traces']) )
        snr    = np.concatenate( (snr,f['snr']) )

    print(f'Loaded {len(traces)-1} traces')

    return traces[1:], snr[1:]


#-#-#- Function from Xishui Tian's transient search -#-#-#
def high_pass_filter(trace, 
                     sample_frequency=500, 
                     cutoff_frequency=50):
    # Nyquist-Shannon sampling theorem: maximum frequency that can be effectively sampled without aliasing when the signal is sampled at a given rate
    Nyquist_frequency = 0.5 * sample_frequency

    # design a Butterworth high-pass filter
    b, a = signal.butter(4, cutoff_frequency / Nyquist_frequency, btype='high', analog=False)
    return signal.filtfilt(b, a, trace)


#-#-#- Function from Xishui Tian's transient search -#-#-#
def search_windows(trace,
                   filter_status='on',
                   threshold1=60,
                   threshold2=36,
                   maxRMS=21,
                   num_cross=2+4,
                   num_separation=750,
                   num_interval=26,
                   sample_frequency=500,
                   cutoff_frequency=50):
    if filter_status == 'on':
        trace = high_pass_filter(trace=trace, sample_frequency=sample_frequency, cutoff_frequency=cutoff_frequency)
 
    # stop if there are no transients/pulses
    cross_threshold1 = np.abs(trace) > threshold1
    cross_threshold2 = np.abs(trace) > threshold2
    # print(f'\n\n\n thresholds {threshold1, threshold2, np.sum(cross_threshold1), np.sum(cross_threshold2), num_cross}')
    if np.sum(cross_threshold1) < 2 or np.sum(cross_threshold2) < num_cross:
        return []

    # exclude abnormal traces
    # if get1trace_RMS(trace=trace) > maxRMS or max(get1trace_PSD(trace=trace)) > maxPSD:
    #     return []
    
    # find trace positions where threshold1 is exceeded
    cross1_ids = np.flatnonzero(cross_threshold1)
    
    # find the separations between consecutive threshold crossings
    cross1_separations = np.diff(cross1_ids)

    # locate pulse indices in threshold crossing indices
    pulse_ids = np.flatnonzero(cross1_separations > num_separation)
    pulse_ids = np.concatenate(([-1], pulse_ids, [len(cross1_ids)-1]))

    window_list = []

    # search all transients/pulses
    for i in range(len(pulse_ids)-1):
        trigger_id = cross1_ids[pulse_ids[i]+1]
        if np.sum(cross_threshold2[trigger_id-num_interval:trigger_id]) > 2:
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
            window_list.append([start_id, stop_id])
        
    return window_list


#-#-#- Function adapted from Xishui Tian's transient search -#-#-#
def thresh_trigger(trace,
                   thresh1=60,
                   thresh2=36):
    
    channels = ['X','Y','Z']
    
    # Search for pulses                                                                                                                                                                                                                                                         
    num_pulse = 0
    list_trigger_flag = [0, 0]

    for i, channel in enumerate(channels[:2]): # exclude channel Z                                                                                                                                                                                                              
        window     = search_windows(trace=trace[i], filter_status='off', threshold1=thresh1, threshold2=thresh2) #trace=high_pass_filter(trace[i])
        num_pulse += len(window)
        if len(window):
            list_trigger_flag[i] = 1

    if list_trigger_flag == [0, 0]:
        # No trigger                                                                                                                                                                                                                                                            
        trigger_flag = 0
    if list_trigger_flag == [1, 0]:
        # X trigger                                                                                                                                                                                                                                                             
        trigger_flag = 1
    if list_trigger_flag == [0, 1]:
        # Y trigger                                                                                                                                                                                                                                                             
        trigger_flag = 2
    if list_trigger_flag == [1, 1]:
        # X Y trigger                                                                                                                                                                                                                                                           
        trigger_flag = 3

    return num_pulse, trigger_flag


def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Processes candidate background pulses to create a background-pulse database.")

    parser.add_argument('input_dir',
                        type=str,
                        help='Input directory containing air-shower pulse traces with added noise from data in npz format.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='./',
                        help='Directory where the npz file will be stored containing the selected background pulses.')
    
    parser.add_argument('-t1',
                        '--thresh1',
                        type=int,
                        default=60,
                        help='Sets the T1 threshold for the threshold trigger in ADC counts.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Load parser arguments -#-#-#
    args       = manage_args()
    input_dir  = args.input_dir
    output_dir = args.output_dir
    thresh1    = args.thresh1

    #-#-#- Load in all signal trace (with noise added) in input directory -#-#-#
    sig_traces, snr = load_sig_traces_with_noise(input_dir)

    #-#-#- Perform the threshold trigger on all traces -#-#-#
    trigger_flags   = np.zeros(len(sig_traces))

    for i, sig_trace in enumerate(sig_traces):
        if (i+1) % 100 == 0:
            print(f'Processing trace {i+1}/{len(sig_traces)}')
        trigger_flags[i] = thresh_trigger(sig_trace,thresh1=thresh1)[1]

    sig_traces_triggered = sig_traces[trigger_flags>0]
    snr_triggered        = snr[trigger_flags>0]

    #-#-#- Save the triggered traces -#-#-#
    rf_chain = input_dir.split('/')[-2].split('_')[0]

    output_filename = f'sig_pulses_gp13_{rf_chain}_th1_{thresh1}.npz'
    output_file     = os.path.join(output_dir,output_filename)

    np.savez(output_file,
             traces=sig_traces_triggered,
             snr=snr_triggered,
             trigger_flags=trigger_flags)
    
    print(f'Saved {len(sig_traces_triggered)} traces in {output_file}')
    print('DONE!')