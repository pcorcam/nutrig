#! /usr/bin/env python3
'''
This file contains the main module for the NUTRIG first-level trigger based on template fitting.
'''

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import glob
import argparse

import numpy as np

import grand.dataio.root_trees as rt

import nutrig.database.tools as tools


###-###-###-###-###-###-###- GENERAL FUNCTIONS -###-###-###-###-###-###-###

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description='Performs the template FLT algorithm on a set of traces.')

    parser.add_argument('sim_dir',
                        type=str,
                        help='Parent directory of GRANDroot files containing simulated air-shower voltages at ADC input.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='/sps/grand/pcorrea/nutrig/template/corr_sim/',
                        help='Directory where the correlation values will be stored.')
    
    parser.add_argument('-t',
                        '--template_dir',
                        type=str,
                        default='/sps/grand/pcorrea/nutrig/template/lib/',
                        help='Directory where the XY templates are stored.')
    
    parser.add_argument('-n',
                        '--n_templates',
                        type=int,
                        default=96,
                        help='Number of templates to consider.')
    
    parser.add_argument('-rf',
                        '--rf_chain',
                        type=str,
                        default='rfv2',
                        help='Version of the RF chain used in the simulations and for the templates.')
    
    parser.add_argument('-p',
                        '--primary',
                        type=str,
                        default='Proton',
                        help='Primary to consider for the simulations.')
    
    parser.add_argument('-th',
                        '--thresh',
                        type=int,
                        default=30,
                        help='ADC threshold for the traces to consider in this script.')
    
    parser.add_argument('-s',
                        '--start_file_idx',
                        type=int,
                        default=0,
                        help='First simulation file to consider (useful for job submission).')
    
    parser.add_argument('-3',
                        '--end_file_idx',
                        type=int,
                        default=None,
                        help='Last simulation file to consider (useful for job submission).')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Get parser arguments -#-#-#
    args           = manage_args()
    sim_dir        = os.path.normpath(args.sim_dir)
    output_dir     = args.output_dir
    template_dir   = args.template_dir
    n_templates    = args.n_templates
    rf_chain       = args.rf_chain
    primary        = args.primary
    thresh         = args.thresh
    start_file_idx = args.start_file_idx
    end_file_idx   = args.end_file_idx

    sim = os.path.basename(sim_dir) # zhaires or coreas
    
    template_file = os.path.join(template_dir,f'templates_{n_templates}_XY_{rf_chain}.npz')
    with np.load(template_file) as f:
        templates = f['templates']
        len_templates = templates.shape[1]

    sim_dir   = os.path.join(sim_dir,f'voltage_{rf_chain}/') #gr_voltage_GP300_Xi_Sib_Proton_3.65_84.5_129.0_4820.root
    sim_files = sorted( glob.glob( os.path.join(sim_dir,f'*_{primary.capitalize()}_*.root') ) )

    if end_file_idx is None:
        #end_file_idx = len(sim_files)
        end_file_idx = start_file_idx + 500 # for jobs, so you take 500 files / job

    print(f'Only condidering files {start_file_idx}-{end_file_idx} in {sim_dir}...')
    sim_files = sim_files[start_file_idx:end_file_idx]

    corr_x = np.zeros( (int(1e7),templates.shape[0]) )
    corr_y = np.zeros( (int(1e7),templates.shape[0]) )

    idx_x = 0
    idx_y = 0

    for sim_number, sim_file in enumerate(sim_files[:]):
        print(f'Processing file {sim_number+1}/{len(sim_files)}...')

        try:
            tvoltage = rt.TVoltage(sim_file)
            tvoltage.get_entry(0)

            trace     = np.array(tvoltage.trace)
            trace     = tools.digitize(trace,adc_sampling_rate=2000) # converts to ADC counts without desampling
            max_trace = np.max( np.abs(trace),axis=2,keepdims=True )

            tvoltage.stop_using()
            tvoltage.close_file()
        except:
            print('Can not open file / no valid trace in file:',sim_file)
            print('Skipping...')
            continue

        entries_above_thresh_x = np.where(max_trace[:,0]>thresh)[0] # find which entries have a maximum above threshold
        entries_above_thresh_y = np.where(max_trace[:,1]>thresh)[0]

        corr_x_new = np.zeros( (entries_above_thresh_x.shape[0],templates.shape[0]) )
        corr_y_new = np.zeros( (entries_above_thresh_y.shape[0],templates.shape[0]) )

        for k, template in enumerate(templates):
            for i, entry in enumerate(entries_above_thresh_x[:]):
                # Only need to compute the correlation at the peak position.
                # Give a little wiggle room just in case (+/- 2.5 ns).
                # Templates are 200 ns by construction (400 samples with simu sampling rate of 2 GHz).
                # The peak in the template occurs 60 ns after the beginning of the trace (120 samples for 2 GHz).
                pos_max           = np.argmax( np.abs( trace[entry,0] ) )
                trace_around_peak = trace[entry,0,pos_max-120-5:pos_max+280+5]

                sliding_trace_window = np.lib.stride_tricks.sliding_window_view(trace_around_peak,len_templates)
                sliding_rms          = tools.rms(sliding_trace_window,samples_from_trace_edge=0,axis=1)

                corr_trace  = np.correlate( trace_around_peak,template,mode='valid' ) / sliding_rms / len_templates
                corr_x_new[i,k] = np.max( np.abs(corr_trace) )
                
            for i, entry in enumerate(entries_above_thresh_y[:]):
                # Just copy paste because lazy to make it cleaner :p 
                pos_max           = np.argmax( np.abs( trace[entry,1] ) )
                trace_around_peak = trace[entry,1,pos_max-120-5:pos_max+280+5]

                sliding_trace_window = np.lib.stride_tricks.sliding_window_view(trace_around_peak,len_templates)
                sliding_rms          = tools.rms(sliding_trace_window,samples_from_trace_edge=0,axis=1)

                corr_trace  = np.correlate( trace_around_peak,template,mode='valid' ) / sliding_rms / len_templates
                corr_y_new[i,k] = np.max( np.abs(corr_trace) )

        corr_x[idx_x:idx_x+len(entries_above_thresh_x)] = corr_x_new
        idx_x += len(entries_above_thresh_x)

        corr_y[idx_y:idx_y+len(entries_above_thresh_y)] = corr_y_new
        idx_y += len(entries_above_thresh_y)

        print(f'File processed! Computed cross correlation for {len(entries_above_thresh_x)} traces in X and {len(entries_above_thresh_y)} traces in Y')

    # Save correlations in one output file
    output_file = os.path.join(output_dir,f'corr_templates_{n_templates}_sim_{sim}_{primary.lower()}_thresh_{thresh}_{rf_chain}_files_{start_file_idx}-{end_file_idx}.npz')
    np.savez(output_file,
             corr_x=corr_x[:idx_x],
             corr_y=corr_y[:idx_y])
    
    print('Saved correlation values at:',output_file)
    print('FINISHED!')