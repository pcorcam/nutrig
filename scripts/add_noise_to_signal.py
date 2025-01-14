###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import glob
import os
import argparse

import numpy as np
from scipy import signal

import grand.dataio.root_trees as rt
from nutrig.simu_analysis.tools import *


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

#-#-#- Copied from GRANDlib's `convert_voltage2adc.py` -#-#-#
def get_noise_traces(data_dir,
                     n_traces,
                     n_files=None,
                     n_samples=2048,
                     rng=np.random.default_rng()):
    '''
    Selects random ADC noise traces from a directory containing files of measured data.

    Arguments
    ---------
    `data_dir`
    type        : str
    description : Path to directory where data files are stored in GrandRoot format.

    `n_traces`
    type        : int
    description : Number of noise traces to select, each with shape (3,n_samples).

    `n_files` (optional)
    type        : int
    description : Number of data files to consider for the selection. Default selects all files.

    `n_samples` (optional)
    type        : int
    description : Number of samples required from the measured data trace.

    `rng` (optional)
    type        : np.random.Generator
    description : Random number generator. Default has an unspecified seed. NOTE: default or seed=None makes the selection irreproducible. 
                                
    Returns
    -------
    `noise_trace`
    type        : np.ndarray[int]
    units       : ADC counts (least significant bits)
    description : The selected array of noise traces, with shape (N_du,3,N_samples).
    '''

    # Select n_files random data files from directory
    data_files = sorted( glob.glob(data_dir+'*.root') ) # sort to get rid of additional randomness of glob

    if n_files is None:
        n_files = len(data_files)

    assert n_files <= len(data_files), f'There are {len(data_files)} in {data_dir} - requested {n_files}'
    idx_files = rng.choice( range( len(data_files) ), n_files, replace=False )
    data_files = [data_files[i] for i in idx_files]

    print(f'Fetching {n_traces} random noise traces of 3 x {n_samples} samples from {n_files} data files in {data_dir}')

    # Reduce files to open if n_traces < n_files
    quotient  = n_files // n_traces
    remainder = n_files % n_traces

    if quotient == 0:
        data_files = data_files[:remainder]
        print(f'Only need to open {remainder} < {n_traces} data files')

    # Get noise traces from data files
    noise_trace = np.empty( (n_traces,3,n_samples),dtype=int )
    trace_idx = 0

    for i, data_file in enumerate(data_files):
        df = rt.DataFile(data_file)
        tadc = df.tadc #rt.TADC(data_file)

        # Check that data traces contain requested number of samples
        tadc.get_entry(0)
        n_samples_data = tadc.adc_samples_count_ch[0][1]*2 #TODO: tempfix

        assert n_samples_data >= n_samples, f'Data trace contains less samples than requested: {n_samples_data} < {n_samples}'

        # Select random entries from TADC
        # NOTE: assumed that each entry corresponds to a single DU with ADC channels (0,1,2)=(X,Y,Z)
        n_entries_tot = tadc.get_number_of_entries()
        
        if i < n_traces % n_files:
            n_entries_sel = n_traces // n_files + 1
        else:
            n_entries_sel = n_traces // n_files
        
        entries_sel = rng.integers(0,high=n_entries_tot,size=n_entries_sel)
        print(f'Selected {n_entries_sel} random traces from {data_file}')
        
        for entry in entries_sel:
            tadc.get_entry(entry)
            trace = np.array(tadc.trace_ch)[0,1:,:n_samples] # toggle [,1:,] to get rid of float channel

            noise_trace[trace_idx] = trace
            trace_idx += 1

        tadc.stop_using()
        tadc.close_file()

    return noise_trace


def add_signal_to_noise(sig_traces,
                        noise_traces,
                        rng=np.random.default_rng(),
                        adc_saturation=2**13):
    
    assert sig_traces.shape == noise_traces.shape

    n_traces  = sig_traces.shape[0]
    n_samples = sig_traces.shape[-1]

    max_sample_sig   = np.argmax(np.abs(sig_traces),axis=2)[:,0] # position same for each polarization
    max_sample_rand  = rng.integers(100,n_samples-100,n_traces) # not near edges of trace
    pulse_start_rand = max_sample_rand - max_sample_sig + n_samples

    sig_plus_noise_traces = np.zeros(sig_traces.shape)
    noise_range_dummy     = np.arange(n_samples,2*n_samples,dtype=int)

    for i in range(n_traces):
        sig_plus_noise_trace = np.zeros((3,n_samples*3))
        sig_range_dummy      = np.arange(pulse_start_rand[i],pulse_start_rand[i]+n_samples,dtype=int)

        sig_plus_noise_trace[:,noise_range_dummy] += noise_traces[i]        
        sig_plus_noise_trace[:,sig_range_dummy]   += sig_traces[i]

        sig_plus_noise_traces[i] = sig_plus_noise_trace[:,noise_range_dummy]

    sig_plus_noise_traces = np.where(np.abs(sig_plus_noise_traces)<adc_saturation,sig_plus_noise_traces,np.sign(sig_plus_noise_traces)*adc_saturation)

    return sig_plus_noise_traces


def filter_traces(traces,
                  freq_bandpass=50,
                  freqs_notch=[119,137],
                  sampling_rate=500):
    '''
    Bandpass filter above > 50 MHz to kill short waves
    Notch filters at 119 MHz and 137 MHz to kill communication lines
    '''

    b, a            = signal.butter(4,freq_bandpass,btype='high',analog=False,fs=sampling_rate)
    traces_filtered = signal.filtfilt(b,a,traces)

    for freq in freqs_notch:
        b,a             = signal.iirnotch(freq,100,fs=sampling_rate)
        traces_filtered = signal.filtfilt(b,a,traces_filtered)

    return np.trunc(traces_filtered)


def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Adds measured noise to ADC traces ")

    parser.add_argument('sig_file',
                        type=str,
                        help='Input file containing air-shower traces in .npz format.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='./',
                        help='Directory where the npz file will be stored with noise added to the input traces.')

    parser.add_argument('-n',
                        '--noise_dir',
                        type=str,
                        default='/sps/grand/data/gp13/GrandRoot/2024/02/',
                        help='Directory where noise traces are stored.')
    
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=0,
                        help='Seed for the random number generator.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Load parser arguments -#-#-#
    args       = manage_args()
    sig_file   = args.sig_file
    output_dir = args.output_dir
    noise_dir  = args.noise_dir
    seed       = args.seed
    
    #-#-#- Load in ADC traces of air showers from input file -#-#-#
    f          = np.load(sig_file)
    sig_traces = f['traces_adc']
    print(f'Loaded {len(sig_traces)} signal traces')

    #-#-#- Randomly select noise traces to add to signal traces -#-#-#
    rng  = np.random.default_rng(seed)
    noise_traces = get_noise_traces(noise_dir,
                                    sig_traces.shape[0],
                                    n_samples=sig_traces.shape[-1],
                                    rng=rng)
    
    #-#-#- Filter both signal and noise traces and add signal to the noise -#-#-#
    sig_traces_filtered   = filter_traces(sig_traces)
    noise_traces_filtered = filter_traces(noise_traces)

    output_traces = add_signal_to_noise(sig_traces_filtered,
                                        noise_traces_filtered,
                                        rng=rng)
    
    #-#-#- Calculate the signal-to-noise ratio -#-#-#
    max_signal = np.max( np.abs(sig_traces_filtered),axis=2 )
    rms_noise  = np.sqrt( np.mean( noise_traces_filtered**2,axis=2 ) )
    snr        = max_signal/rms_noise

    #-#-#- Save the signal+noise traces -#-#-#
    output_filename = sig_file.split('/')[-1].replace('sig_traces_adc','sig_traces_with_noise').replace('.npz',f'_seed_{seed}.npz')
    output_file     = os.path.join(output_dir,output_filename)

    np.savez(output_file,
             traces=output_traces,
             snr=snr)
    
    print(f'Saved {len(sig_traces)} signal+noise traces in {output_file}')
    print('DONE!')