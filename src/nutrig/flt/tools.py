#! /usr/bin/env python3

'''

'''

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import glob
import sys
import logging

import numpy as np

from database.tools import get_snr

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


def load_results(result_file,
                 n_pol):
    
    results = {}

    with np.load(result_file) as f:
        results['comp_time']         = f['comp_time']
        results['idx_template_best'] = f['idx_template_best']
        results['ts']                = f['ts']
        results['corr_best_all']     = f['corr_best']
        results['time_best_all']     = f['time_best']
        results['ampl_best_all']     = f['ampl_best']
        results['chi2_all']          = f['chi2']
        results['rss_post_peak_all'] = f['rss_post_peak']

        try:
            mask_pol = f['mask_pol']
        except:
            mask_pol = np.ones(results['ts'].shape,dtype=bool)
        
    n_average = np.sum(mask_pol,axis=1) # the denominator for the average depends on how many channels are treated per event

    results['corr_max']  = np.zeros(results['corr_best_all'].shape[0])
    results['corr_mean'] = np.zeros(results['corr_best_all'].shape[0])
    results['corr_prod'] = np.ones(results['corr_best_all'].shape[0])
    results['chi2_min']  = np.zeros(results['corr_best_all'].shape[0])
    results['rss_max']   = np.zeros(results['corr_best_all'].shape[0])
    results['time_best'] = np.zeros(results['corr_best_all'].shape[0])
    results['idx_pol']   = np.zeros(results['corr_best_all'].shape[0])

    for i in range(n_pol):
        corr_best_pol = np.abs( np.take_along_axis( results['corr_best_all'][:,i],results['idx_template_best'][:,i,None],axis=1 )[:,0] )
        chi2_best_pol = np.take_along_axis( results['chi2_all'][:,i],results['idx_template_best'][:,i,None],axis=1 )[:,0]
        rss_max_pol   = np.max( results['rss_post_peak_all'][:,i],axis=1 )
        time_best_pol = np.take_along_axis( results['time_best_all'][:,i],results['idx_template_best'][:,i,None],axis=1 )[:,0]

        results['time_best']  = np.where(corr_best_pol>results['corr_max'],time_best_pol,results['time_best'])
        results['idx_pol']    = np.where(corr_best_pol>results['corr_max'],i,results['idx_pol'])
        results['corr_max']   = np.where(corr_best_pol>results['corr_max'],corr_best_pol,results['corr_max'])
        results['corr_mean'] += corr_best_pol/n_average

        results['corr_prod'][mask_pol[:,i]] *= corr_best_pol[mask_pol[:,i]]
        results['chi2_min'][mask_pol[:,i]]   = np.where(-1*chi2_best_pol[mask_pol[:,i]]>-1*results['chi2_min'][mask_pol[:,i]],results['chi2_min'][mask_pol[:,i]],chi2_best_pol[mask_pol[:,i]])
        results['rss_max'][mask_pol[:,i]]    = np.where(rss_max_pol[mask_pol[:,i]]>results['rss_max'][mask_pol[:,i]],rss_max_pol[mask_pol[:,i]],results['rss_max'][mask_pol[:,i]])

    return results


def get_snr_masks(sig_dataset_file,
                  snr_bin_width=1,
                  snr_min=3,
                  snr_max=7):

    with np.load(sig_dataset_file) as f:
        sig_snr = get_snr(f['traces'],f['inj_pulse_times'])
        sig_snr = np.max(sig_snr[:,:2],axis=1) # Take the largest SNR of X and Y as the SNR parameter

    snr_bin_edges = np.arange(snr_min,snr_max+snr_bin_width,snr_bin_width)
    snr_masks     = {snr_bin : None for snr_bin in snr_bin_edges[:-1]}

    for snr_bin in snr_masks.keys():
        snr_masks[snr_bin] = np.where( np.logical_and( sig_snr>=snr_bin, sig_snr<snr_bin+snr_bin_width ), True, False )

    return snr_masks


def get_selection_efficiency(ts_values,
                             ts_thresh):
    
    selection_eff = np.zeros(ts_thresh.shape)

    for i, ts in enumerate(ts_thresh):
        sel_mask         = np.where(ts_values>=ts,True,False)
        selection_eff[i] = len(sel_mask[sel_mask])

    selection_eff /= len(ts_values)

    return selection_eff