#! /usr/bin/env python3
'''
This file contains the main module for the NUTRIG first-level trigger based on template fitting.
'''

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import glob
import time
import logging
import argparse

import numpy as np

import nutrig.flt.tools as tools

from nutrig.flt.template_FLT import *

logger = logging.getLogger(__name__)


# ###-###-###-###-###-###-###- LOGGER SETUP -###-###-###-###-###-###-###

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)


###-###-###-###-###-###-###- GENERAL FUNCTIONS -###-###-###-###-###-###-###

def get_mask_pol(traces,
                 pretrig_flags):
    
    max_traces = np.max( np.abs(traces),axis=2 )
    max_pol    = np.argmax( max_traces[...,:2],axis=1 ) # exclude Z

    mask_pol = np.zeros(pretrig_flags.shape,dtype=int)      # all X = channel 0
    mask_pol = np.where(pretrig_flags==2,1,mask_pol)        # add only Y = channel 1
    mask_pol = np.where(pretrig_flags==12,max_pol,mask_pol) # add X+Y = max of channel 0 or 1

    return mask_pol


def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description='Performs the template FLT algorithm on a set of traces.')

    parser.add_argument('input_file',
                        type=str,
                        help='Input file in npz format containing traces with pre-trigger info.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='/sps/grand/pcorrea/nutrig/template/results/',
                        help='Directory where the FLT results will be stored.')
    
    parser.add_argument('-t',
                        '--template_dir',
                        type=str,
                        default='/sps/grand/pcorrea/nutrig/template/lib/',
                        help='Directory where templates are stored in npz format.')
    
    parser.add_argument('-n',
                        '--n_templates',
                        type=int,
                        default=96,
                        help='Number of templates.')
    
    parser.add_argument('-p',
                        '--pol',
                        type=str,
                        default='XY',
                        help='Polarizations of the templates.')
    
    parser.add_argument('-rf',
                        '--rf_chain',
                        type=str,
                        default='rfv2',
                        help='RF chain used to generate the templates.')
    
    parser.add_argument('-cw',
                        '--corr_window',
                        type=int,
                        nargs=2,
                        default=None,
                        help='Time window for correlation relative to pre-trigger time. Unit: ADC samples.')
    
    parser.add_argument('-fw',
                        '--fit_window',
                        type=int,
                        nargs=2,
                        default=None,
                        help='Time window for fitting relative to best-correlation time. Unit: ADC samples.')
    
    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Get parser arguments -#-#-#
    args            = manage_args()
    input_file      = args.input_file
    output_dir      = args.output_dir
    template_dir    = args.template_dir
    n_templates     = args.n_templates
    pol             = args.pol
    rf_chain        = args.rf_chain
    corr_window     = args.corr_window
    fit_window      = args.fit_window
    verbose         = args.verbose

    logger = tools.load_logger('debug')
    logger.info('*** START OF SCRIPT ***')
    
    #-#-#- Load traces from input file -#-#-#
    with np.load(input_file) as f:
        traces        = f['traces']
        pretrig_flags = f['pretrig_flags']
        pretrig_times = f['pretrig_times']

    n_traces = len(traces)

    #-#-#- Initiate and set up template FLT -#-#-#
    FLT = TemplateFLT()
    FLT.load_templates(template_dir,n_templates,pol,rf_chain)

    if corr_window is not None:
        FLT.set_corr_window(corr_window)
    if fit_window is not None:
        FLT.set_fit_window(fit_window)

    #-#-#- Find for which polarization you want to perform the FLT -#-#-#
    mask_pol = get_mask_pol(traces,pretrig_flags)

    #-#-#- Initiate FLT info to save -#-#-#
    comp_time         = np.zeros( (n_traces,) )
    idx_template_best = np.zeros( (n_traces,),dtype=int )
    ts                = np.zeros( (n_traces,) )
    corr_best         = np.zeros( (n_traces,FLT.n_templates) )
    time_best         = np.zeros( (n_traces,FLT.n_templates),dtype=int )
    ampl_best         = np.zeros( (n_traces,FLT.n_templates) )
    chi2              = np.zeros( (n_traces,FLT.n_templates) )
    rss_post_peak     = np.zeros( (n_traces,FLT.n_templates) )
    
    #-#-#- Perform the template FLT for all traces -#-#-#
    for i, trace in enumerate(traces[:]):
        logger.info(f'Processing trace {i+1}/{n_traces}')

        t0 = time.time()
        FLT.template_fit( trace[ mask_pol[i] ], pretrig_times[i] ) # only consider 1 dimension
        
        comp_time[i]         = time.time() - t0
        idx_template_best[i] = FLT.idx_template_best_fit
        ts[i]                = FLT.ts
        corr_best[i]         = FLT.corr_best
        time_best[i]         = FLT.time_best
        ampl_best[i]         = FLT.ampl_best
        chi2[i]              = FLT.chi2
        rss_post_peak[i]     = FLT.rss_post_peak

    #-#-#- Save template FLT results -#-#-#
    output_filename  = 'results_template_FLT_'
    output_filename += os.path.splitext( os.path.basename(input_file) )[0]
    output_filename += f'_templates_{FLT.n_templates}_cw_{FLT._corr_window[0]}_{FLT._corr_window[1]}_fw_{FLT._fit_window[0]}_{FLT._fit_window[1]}.npz'

    output_file = os.path.join(output_dir,output_filename)
    np.savez(output_file,
             comp_time=comp_time,
             idx_template_best=idx_template_best,
             ts=ts,
             corr_best=corr_best,
             time_best=time_best,
             ampl_best=ampl_best,
             chi2=chi2,
             rss_post_peak=rss_post_peak,
             mask_pol=mask_pol)
    logger.info(f'Saved template FLT results in {output_file}')

    logger.info('*** END OF SCRIPT ***')


    