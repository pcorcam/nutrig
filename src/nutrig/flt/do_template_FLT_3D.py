#! /usr/bin/env python3
'''
This file contains the main module for the NUTRIG first-level trigger based on template fitting.
'''

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import time
import logging
import argparse

import numpy as np

import tools

from template_FLT import *

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
    
    parser.add_argument('-nx',
                        '--n_templates_X',
                        type=int,
                        default=96,
                        help='Number of templates to load for X.')
    
    parser.add_argument('-ny',
                        '--n_templates_Y',
                        type=int,
                        default=96,
                        help='Number of templates to load for Y.')
    
    parser.add_argument('-nz',
                        '--n_templates_Z',
                        type=int,
                        default=None,
                        help='Number of templates to load for Z.')

    parser.add_argument('-r',
                        '--random_temp',
                        type=int,
                        default=None,
                        help='Index of template library where the templates were chosen randomly.')
    
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
    n_templates_X   = args.n_templates_X
    n_templates_Y   = args.n_templates_Y
    n_templates_Z   = args.n_templates_Z
    random_temp     = args.random_temp
    rf_chain        = args.rf_chain
    corr_window     = args.corr_window
    fit_window      = args.fit_window
    verbose         = args.verbose

    logger = tools.load_logger(verbose)
    logger.info('*** START OF SCRIPT ***')
    
    #-#-#- Load traces from input file -#-#-#
    with np.load(input_file) as f:
        traces        = f['traces']
        pretrig_flags = f['pretrig_flags']
        pretrig_times = f['pretrig_times']

    n_traces = len(traces)

    #-#-#- Obtain masks for each channel -#-#-#

    #-#-#- Initiate and set up template FLT -#-#-#
    FLT = TemplateFLT3D()
    pol = ''
    n_templates = []

    if n_templates_X is not None:
        FLT.load_templates_X(template_dir,n_templates_X,rf_chain,random_temp=random_temp)
        pol += 'X'
        n_templates.append(n_templates_X)
        logger.info('Loaded templates for X')

    if n_templates_Y is not None:
        FLT.load_templates_Y(template_dir,n_templates_Y,rf_chain,random_temp=random_temp)
        pol += 'Y'
        n_templates.append(n_templates_Y)
        logger.info('Loaded templates for Y')

    if n_templates_Z is not None:
        FLT.load_templates_Z(template_dir,n_templates_Z,rf_chain,random_temp=random_temp)
        pol += 'Z'
        n_templates.append(n_templates_Z)
        logger.info('Loaded templates for Z')

    n_pol           = len(n_templates) # to only save info for polarizations that we care for
    n_templates_max = max(n_templates) # to mitigate different number of templates

    if corr_window is not None:
        FLT.set_corr_window(corr_window)
    if fit_window is not None:
        FLT.set_fit_window(fit_window)

    #-#-#- Initiate FLT info to save -#-#-#
    comp_time         = np.zeros( (n_traces,) )
    idx_template_best = np.zeros( (n_traces,n_pol),dtype=int )
    ts                = np.zeros( (n_traces,n_pol) )
    corr_best         = np.zeros( (n_traces,n_pol,n_templates_max) )
    time_best         = np.zeros( (n_traces,n_pol,n_templates_max),dtype=int )
    ampl_best         = np.zeros( (n_traces,n_pol,n_templates_max) )
    chi2              = np.zeros( (n_traces,n_pol,n_templates_max) )
    rss_post_peak     = np.zeros( (n_traces,n_pol,n_templates_max) )
    mask_pol          = np.zeros( (n_traces,n_pol),dtype=bool )
    
    #-#-#- Perform the template FLT for all traces -#-#-#
    for i, trace in enumerate(traces[:]):
        logger.info(f'Processing trace {i+1}/{n_traces}')

        t0 = time.time()
        FLT.template_fit( trace, pretrig_times[i], pretrig_flags[i] ) # it will only fit on channels with a pretrigger flag
        
        comp_time[i] = time.time() - t0

        j = 0
        if 'X' in pol:
            if pretrig_flags[i] in [1,12,13,123]:
                idx_template_best[i,j] = FLT.FLT_X.idx_template_best_fit
                ts[i,j]                = FLT.FLT_X.ts
                corr_best[i,j]         = FLT.FLT_X.corr_best
                time_best[i,j]         = FLT.FLT_X.time_best
                ampl_best[i,j]         = FLT.FLT_X.ampl_best
                chi2[i,j]              = FLT.FLT_X.chi2
                rss_post_peak[i,j]     = FLT.FLT_X.rss_post_peak
                mask_pol[i,j]          = True
            j += 1

        if 'Y' in pol:
            if pretrig_flags[i] in [2,12,23,123]:
                idx_template_best[i,j] = FLT.FLT_Y.idx_template_best_fit
                ts[i,j]                = FLT.FLT_Y.ts
                corr_best[i,j]         = FLT.FLT_Y.corr_best
                time_best[i,j]         = FLT.FLT_Y.time_best
                ampl_best[i,j]         = FLT.FLT_Y.ampl_best
                chi2[i,j]              = FLT.FLT_Y.chi2
                rss_post_peak[i,j]     = FLT.FLT_Y.rss_post_peak
                mask_pol[i,j]          = True
            j += 1

        if 'Z' in pol:
            if pretrig_flags[i] in [3,13,23,123]:
                idx_template_best[i,j] = FLT.FLT_Z.idx_template_best_fit
                ts[i,j]                = FLT.FLT_Z.ts
                corr_best[i,j]         = FLT.FLT_Z.corr_best
                time_best[i,j]         = FLT.FLT_Z.time_best
                ampl_best[i,j]         = FLT.FLT_Z.ampl_best
                chi2[i,j]              = FLT.FLT_Z.chi2
                rss_post_peak[i,j]     = FLT.FLT_Z.rss_post_peak
                mask_pol[i,j]          = True


    #-#-#- Save template FLT results -#-#-#
    output_filename  = f'results_template_FLT_{pol}_'
    output_filename += os.path.splitext( os.path.basename(input_file) )[0]
    output_filename += f'_templates'
    if random_temp is not None:
        output_filename += f'_random_{random_temp}'
    output_filename += f'_{n_templates_max}_cw_{FLT.FLT_X._corr_window[0]}_{FLT.FLT_X._corr_window[1]}_fw_{FLT.FLT_X._fit_window[0]}_{FLT.FLT_X._fit_window[1]}.npz'

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