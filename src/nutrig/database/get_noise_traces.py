###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import glob
import os
import argparse

import numpy as np
from scipy import signal

import grand.dataio.root_trees as rt # type: ignore

import tools as tools


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def get_root_files(data_dir):

    root_files = glob.glob( os.path.join(data_dir,'*.root') )

    return sorted(root_files)


def get_gp13_traces(root_file,
                    n_samples=1024,
                    float_ch=0):
    
    channels = np.arange(4)
    mask_ch  = np.delete(channels,float_ch)
    
    df   = rt.DataFile(root_file)
    tadc = df.tadc
    #tadc = rt.TADC(root_file)

    n_traces  = tadc.get_number_of_entries()

    traces = np.zeros( (n_traces,3,n_samples) )

    for i in range(n_traces):
        tadc.get_entry(i)
        trace           = np.array(tadc.trace_ch[0])
        n_samples_trace = trace.shape[-1]

        if n_samples_trace >= n_samples:
            traces[i] = trace[mask_ch,:n_samples]
        else:
            traces[i,:,:n_samples_trace] = trace[mask_ch]

    df.close()

    return traces


def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Gets noise traces from data and stores them in npz files.")

    parser.add_argument('input_dir',
                        type=str,
                        help='Input directory containing data files in GrandRoot format.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='./',
                        help='Directory where the noise traces will be stored in npz format.')
    
    parser.add_argument('-ns',
                        '--n_samples',
                        type=int,
                        default=1024,
                        help='Number of samples to save of the traces.')

    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Load parser arguments -#-#-#
    args       = manage_args()
    input_dir  = args.input_dir
    output_dir = args.output_dir
    n_samples  = args.n_samples

    #-#-#- Get all GrandRoot files in input directory -#-#-#
    root_files = get_root_files(input_dir)
    print(f'Processing {len(root_files)} GrandRoot files in {input_dir}')

    #-#-#- Load in traces for each GrandRoot file and save in output directory in npz format -#-#-#
    for i, root_file in enumerate(root_files[340:341]): #340 + 33
        print(f'Processing file {i+1}/{len(root_files)}')

        traces = get_gp13_traces(root_file,n_samples=n_samples)
        traces_filtered = tools.filter_traces(traces)

        root_filename = os.path.basename(root_file)
        npz_filename  = root_filename.replace('.root','.npz')
        npz_file      = os.path.join(output_dir,npz_filename)

        np.savez(npz_file,traces=traces_filtered)
        
        print(f'Saved {traces.shape[0]} traces with a shape {traces.shape[1:]} in {npz_file}')

    print('FINISHED')

###-###-###-###-###-###-###- END -###-###-###-###-###-###-###