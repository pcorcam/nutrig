#! /usr/bin/env python3
'''
Constructs a database of background pulses for trigger studies. 

The start point are candidate transient pulses in GP13 data found by Xishui Tian's analysis; see /sps/grand/xtian/transient_search_scripts

The input candidate pulses are then checked to fulfill the following criteria:
- the traces need to be clean, i.e. no chopped events
- the pulse cannot be too close to the edges of the trace (default = 100 samples)

The traces in the database will have a length of 1024 samples by default.
If the datafiles contain traces of longer length, these will be reduced to a window centered around the pulse.

TO RUN:
    python make_bkg_pulse_database.py <input_dir> -o <output_dir>
'''


###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import glob
import os
import argparse

import numpy as np
from scipy import signal

import grand.dataio.root_trees as rt


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def read_txt_files(txt_dir):
    '''
    Reads in the text file output from Xishui Tian's transient analysis.

    Arguments
    ---------
    `txt_dir`
    type        : str
    description : Directory where the output files are stored in .txt format.

                                
    Returns
    -------
    `file_contents`
    type        : dict
    description : Dictionary containing all relevant information of the text files.
    '''

    # Define relevant fields to load in from text files
    file_contents = {'root_files':[],
                     'du_seconds':[],
                     'du_ids':[],
                     'months':[],
                     'tags':[],
                     'entries':[]}
    
    # Loop over text files and load in their info
    txt_files = sorted( glob.glob(txt_dir+'*.txt') )

    for txt_file in txt_files:
        f = open(txt_file,'r')

        for line in f:
            if line[0] == '#':
                continue

            file_contents['root_files'].append( txt_file.split('/')[-1].replace('.root.trigger.txt','.root') )
            file_contents['du_seconds'].append( int(line.split()[3]) )
            file_contents['du_ids'].append( int(line.split()[4]) )
            file_contents['months'].append( txt_file.split('/')[-1].split('_')[1][4:6] )
            file_contents['tags'].append( int(line.split()[2]) )
            file_contents['entries'].append( int(line.split()[0]) )

        f.close()

    return file_contents


def pass_check_rms_float(trace,
                         rms_thresh=10):
    '''
    Checks if the RMS in the floating channel is not too high at the beginning
    or end of the trace. This occurs when there are chopped events in the data.
    NOTE: it is assumed that floating channel = 0.

    Arguments
    ---------
    `trace`
    type        : np.ndarray[int]
    units       : ADC counts (least significant bits)
    description : The trace to check, with shape (N_channels,N_samples).

    `rms_thresh` (optional)
    type        : int
    units       : ADC counts (least significant bits)
    description : The RMS threshold that cannot be exceeded in the floating channel.

                                
    Returns
    -------
    `res`
    type        : bool
    description : True if the check is passed.
    '''

    res = True

    rms_float_start = np.sqrt( np.mean( trace[0][:100]**2 ) )
    rms_float_end   = np.sqrt( np.mean( trace[0][-100:]**2 ) )

    # Check if RMS is too high at beginning or end of the trace (for float a typical value is RMS=5)
    if rms_float_start > rms_thresh or rms_float_end > rms_thresh:
        print('WARNING: RMS threshold exceeded in float channel! Likely a chopped event!')
        print(f'RMS start = {rms_float_start}, RMS end = {rms_float_end}')
        res = False

    return res


def pulse_near_edge(trace,
                    tag,
                    min_sep=100):
    '''
    Checks if the pulse position occurs within `min_sep` ADC samples (1 sample = 2 ns) from each trace edge.
    This is to ensure the pulse is well-contained, and to reject glitches that can appear at the trace edges. 
    NOTE: the pulse position is assumed to correspond to the trace maximum.

    Arguments
    ---------
    `trace`
    type        : np.ndarray[int]
    units       : ADC counts (least significant bits)
    description : The trace to check, with shape (N_channels,N_samples).

    `tag`
    type        : int
    description : Tag corresponding to the polarization information.

    `min_sep` (optional)
    type        : int
    units       : ADC samples
    description : The minimum separation allowed between the pulse and the edge of the trace.

                                
    Returns
    -------
    `res`
    type        : bool
    description : True if the pulse is too close to the edge.
    '''
    
    res        = False
    sample_max = np.argmax(np.abs(trace),axis=1)
    
    # Check for X polarization
    if tag == 1 or tag == 3:
        if sample_max[1] < min_sep or sample_max[1] > trace.shape[-1] - min_sep:
            res = True
            print(f'WARNING: pulse position to close to trace edge! Pulse position = {sample_max[1]}')

    # Check for Y polarization
    if tag == 2 or tag == 3:
        if sample_max[2] < min_sep or sample_max[2] > trace.shape[-1] - min_sep:
            res = True
            print(f'WARNING: pulse position to close to trace edge! Pulse position = {sample_max[2]}')
    
    return res


def reduce_trace_size(trace,
                      tag,
                      len_reduced_trace):
    '''
    Reduces the length of a trace to `len_reduced_trace` (e.g. from 4096 to 1024 samples).
    Ensures that a pulse is not near the edge of the trace.
    NOTE: the pulse position is assumed to correspond to the trace maximum.

    Arguments
    ---------
    `trace`
    type        : np.ndarray[int]
    units       : ADC counts (least significant bits)
    description : The trace to process, with shape (N_channels,N_samples).

    `tag`
    type        : int
    description : Tag corresponding to the polarization information.

    `len_reduced_trace`
    type        : int
    units       : ADC samples
    description : The length of the reduced trace.

                                
    Returns
    -------
    `res`
    type        : bool
    description : The trace with a reduced length, with shape (N_channels,len_reduced_trace)
    '''
    
    assert len_reduced_trace % 1024 == 0, 'New trace length needs to be a multiple of 1024'

    # Find the pulse position
    sample_max = np.argmax(np.abs(trace),axis=1)

    if tag == 1 or tag == 3:
        sample_max = sample_max[1]
    else:
        sample_max = sample_max[2]

    # Set a window for the short trace centered around the pulse 
    window_reduced_trace  = np.arange(len_reduced_trace)
    window_reduced_trace += int( sample_max - len_reduced_trace/2 )

    # Shift the window if it's outside the original trace bounds
    while window_reduced_trace[0] < 0:
        window_reduced_trace += 10
    while window_reduced_trace[-1] >= trace.shape[-1]:
        window_reduced_trace -= 10

    reduced_trace = trace[:,window_reduced_trace]

    print(f'Shortened trace from {trace.shape[-1]} to {len_reduced_trace} samples')

    return reduced_trace


def load_traces(root_files,
                entries,
                months,
                tags,
                target_trace_length=1024):
    '''
    Main function to construct a database of background pulses with trace length `target_trace_length`.
    Loads good-quality traces from data files containing candidate transient pulses.
    NOTE: it is assumed that the data comes from GP13.

    Arguments
    ---------
    `root_files`
    type        : list[str]
    description : Names of GrandRoot files containing the candidate transient events.

    `entries`
    type        : list[int]
    description : The entries of the corresponding GrandRoot files containing the candidate transient events.

    `months`
    type        : list[str]
    description : Month in which the data was taken. Format is MM.

    `tags`
    type        : list[int]
    description : Tags corresponding to the polarization where a candidate pulse was recorded: 1 = X, 2 = Y, 3 = XY.

    `target_trace_length`
    type        : int
    units       : ADC samples
    description : The targeted length of traces in the background-pulse database.

                                
    Returns
    -------
    `traces`
    type        : np.ndarray[int]
    description : The selected background-pulse traces, with shape (N_selected,3,target_trace_length).
    '''
    
    # Prepare trace arrays and mask to filter out bad traces
    n_files            = len(root_files)
    traces             = np.zeros((n_files,3,target_trace_length))
    len_original_trace = np.zeros(n_files)
    mask               = np.ones(n_files,dtype=bool)

    # Prepare filter as done in Xishui Tian's transient analysis
    b_filter, a_filter = signal.butter(4,50/250,btype='high',analog=False)

    # Load in traces
    skip_next = False
    for i, root_file, entry, month, tag in zip(range(n_files),root_files,entries,months,tags):

        if skip_next:
            skip_next = False
            continue

        data_dir = f'/sps/grand/data/gp13/GrandRoot/2024/{month}/'

        tadc = rt.TADC(os.path.join(data_dir,root_file))
        tadc.get_entry(entry)

        print(f'Analyzing pulse {i+1}/{len(root_files)}')

        # Get trace and check that it's not a chopped event
        trace                 = np.array(tadc.trace_ch[0],dtype=int)
        len_original_trace[i] = trace.shape[-1]

        if not pass_check_rms_float(trace):
            mask[i] = False

            tadc.stop_using()
            tadc.close_file()

            print('>> skipping...')
            continue

        # Filter the trace and check that the pulse is not to close near the edge of the trace
        trace_filtered = np.trunc( signal.filtfilt(b_filter,a_filter,trace) )

        if pulse_near_edge(trace_filtered,tag):
            mask[i] = False

            tadc.stop_using()
            tadc.close_file()

            print('>> skipping...')
            continue

        # Shorten the trace if necessary
        if trace_filtered.shape[-1] > target_trace_length:
            trace_filtered = reduce_trace_size(trace_filtered,tag,target_trace_length)

        # Save the trace without the floating channel
        traces[i] = trace_filtered[1:,:target_trace_length]

        tadc.stop_using()
        tadc.close_file()

    return traces[mask], len_original_trace[mask]


def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Processes candidate background pulses to create a background-pulse database.")

    parser.add_argument('input_dir',
                        type=str,
                        help='Input directory containing the transient analysis results of Xishui Tian.\
                              Files must be in .txt format.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='./',
                        help='Directory where the npz file will be stored containing the selected background pulses.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Load parser arguments -#-#-#
    args       = manage_args()
    input_dir  = args.input_dir
    output_dir = args.output_dir

    #-#-#- Load in transient analysis files -#-#-#
    file_contents = read_txt_files(input_dir)

    #-#-#- Load in traces for the background-pulse database -#-#-#
    traces, len_original_trace = load_traces(file_contents['root_files'],
                                             file_contents['entries'],
                                             file_contents['months'],
                                             file_contents['tags'])
    
    #-#-#- Save the selected background pulses -#-#-#
    output_file = 'bkg_' + input_dir.split('/')[-2] + '.npz'
    np.savez(os.path.join(output_dir,output_file),
             traces=traces,
             len_original_trace=len_original_trace)

    print(f'FINISHED! Saved {len(traces)} pulses')
