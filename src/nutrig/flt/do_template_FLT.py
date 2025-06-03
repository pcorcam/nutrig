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

import flt.tools as tools

from template_FLT import *

logger = logging.getLogger(__name__)


####################
# FIXED PARAMETERS #
####################

SIM_SOFTWARE     = 'ZHAireS'
SIM_TAG          = 'DC2.1rc2'
TEMPLATE_DIR     = '/sps/grand/pcorrea/nutrig/template/v2/lib'
THRESH           = 100
N_TEMPLATES      = 5
FILTER_TAG       = 'FILTERED'
TEMPLATE_FILE_XY = os.path.join(TEMPLATE_DIR,f'templates_{SIM_SOFTWARE}_{SIM_TAG}_{FILTER_TAG}_XY_THRESH_{THRESH}_{N_TEMPLATES}.npz')
TEMPLATE_FILE_Z  = os.path.join(TEMPLATE_DIR,f'templates_{SIM_SOFTWARE}_{SIM_TAG}_{FILTER_TAG}_Z_THRESH_{THRESH}_{N_TEMPLATES}.npz')

MODE_SNR     = 'UNIFORM'
BKG_DIR      = '/sps/grand/pcorrea/nutrig/database/v2/bkg/lib'
BKG_FILE     = os.path.join(BKG_DIR,f'bkg_database_nutrig_v2_{FILTER_TAG}_{MODE_SNR}.npz')
SIG_FILE     = BKG_FILE.replace('bkg','sig')

CH_TEMPLATE_FLT  = [0,1]
POL_TEMPLATE_FLT = {0:'X',1:'Y',2:'Z'}
CH_TEMPLATE_FLT_STR = ''
for ch in CH_TEMPLATE_FLT:
    CH_TEMPLATE_FLT_STR += POL_TEMPLATE_FLT[ch]

CORR_WINDOW       = [-10,10]
SIM_SAMPLING_RATE = 500 
ADC_SAMPLING_RATE = 500

OUT_DIR = '/sps/grand/pcorrea/nutrig/template/v2/results'
OUT_FILE_BKG = os.path.join(OUT_DIR,f'bkg_results_{FILTER_TAG}_{MODE_SNR}_template_FLT_CH_{CH_TEMPLATE_FLT_STR}.npz')
OUT_FILE_SIG = OUT_FILE_BKG.replace('bkg','sig')


#############
# FUNCTIONS #
#############

def load_data(nutrig_database_file):

    data = {}

    with np.load(nutrig_database_file) as f:
        data['traces']     = f['traces']
        data['FLT0_flags'] = f['FLT0_flags']
        data['snr']        = f['snr']
        data['t_pulse']    = f['t_pulse']

    return data


def do_template_FLT(data,
                    template_FLT_dict,
                    ch_template_FLT):
    
    res = {}

    res['corr_max']          = np.zeros( ( data['traces'].shape[0] ), dtype=float )
    res['time_best']         = np.zeros( ( data['traces'].shape[0] ), dtype=int )
    res['idx_template_best'] = np.zeros( ( data['traces'].shape[0] ), dtype=int )

    for i in range(data['traces'].shape[0]):
        corr_event              = np.zeros( len(ch_template_FLT),dtype=float )
        idx_template_best_event = np.zeros( len(ch_template_FLT),dtype=int )
        time_best_event         = np.zeros( len(ch_template_FLT),dtype=int )

        # Loop over all target channels
        for ch in ch_template_FLT:
            # Make sure that template FLT was initialized correctly
            if template_FLT_dict[ch] == None:
                continue

            # Perform the template fit
            if data['FLT0_flags'][i,ch]:

                template_FLT_dict[ch].template_fit(data['traces'][i,ch],data['t_pulse'][i,ch])

                corr_event[ch]              = template_FLT_dict[ch].ts
                idx_template_best_event[ch] = template_FLT_dict[ch].idx_template_best_fit
                time_best_event[ch]         = template_FLT_dict[ch].time_best[idx_template_best_event[ch]]

        best_idx_pol = np.argmax(corr_event)

        res['corr_max'][i]          = corr_event[best_idx_pol]
        res['idx_template_best'][i] = idx_template_best_event[best_idx_pol]
        res['time_best'][i]         = time_best_event[best_idx_pol]

    return res


###########################
# DEFINE PARSER ARGUMENTS #
###########################

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Find background pulses in a GP80 MD data file.")

    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')
    
    return parser.parse_args()


################
# MAIN PROGRAM #
################

if __name__ == '__main__':
    parse_args = manage_args()

    logger = tools.load_logger(parse_args.verbose)
    logger.info('*** START OF SCRIPT ***')

    # Load in NUTRIG database
    logger.info(f'Loading background database: {BKG_FILE}')
    try:
        bkg_data = load_data(BKG_FILE)
    except:
        logger.warning('No background database provided! Skipping...')
        bkg_data = None

    logger.info(f'Loading signal database: {SIG_FILE}')
    try:
        sig_data = load_data(SIG_FILE)
    except:
        logger.warning('No signal database provided! Skipping...')
        sig_data = None



    # Set up template FLT objects
    template_FLT_dict = {ch : None for ch in CH_TEMPLATE_FLT}

    for ch in CH_TEMPLATE_FLT:
        if not ch in [0,1,2]:
            logger.warning(f'Not a valid channel: {ch}')
            continue

        if POL_TEMPLATE_FLT[ch] == 'X':
            try:
                template_FLT_X = TemplateFLT()
                template_FLT_X.set_sampling_rates(ADC_SAMPLING_RATE,SIM_SAMPLING_RATE)
                template_FLT_X.set_corr_window(CORR_WINDOW)
                template_FLT_X.load_templates(TEMPLATE_FILE_XY)
                template_FLT_dict[ch] = template_FLT_X
            except Exception as e:
                logger.warning(f'Could not load template FLT for {POL_TEMPLATE_FLT[ch]}: {e}')
                logger.warning(f'SKIPPING template FLT evaluation for {POL_TEMPLATE_FLT[ch]}!')

        if POL_TEMPLATE_FLT[ch] == 'Y':
            try:
                template_FLT_Y = TemplateFLT()
                template_FLT_Y.set_sampling_rates(ADC_SAMPLING_RATE,SIM_SAMPLING_RATE)
                template_FLT_Y.set_corr_window(CORR_WINDOW)
                template_FLT_Y.load_templates(TEMPLATE_FILE_XY)
                template_FLT_dict[ch] = template_FLT_Y
            except Exception as e:
                logger.warning(f'Could not load template FLT for {POL_TEMPLATE_FLT[ch]}: {e}')
                logger.warning(f'SKIPPING template FLT evaluation for {POL_TEMPLATE_FLT[ch]}!')

        if POL_TEMPLATE_FLT[ch] == 'Z':
            try:
                template_FLT_Z = TemplateFLT()
                template_FLT_Z.set_sampling_rates(ADC_SAMPLING_RATE,SIM_SAMPLING_RATE)
                template_FLT_Z.set_corr_window(CORR_WINDOW)
                template_FLT_Z.load_templates(TEMPLATE_FILE_Z)
                template_FLT_dict[ch] = template_FLT_Z
            except Exception as e:
                logger.warning(f'Could not load template FLT for {POL_TEMPLATE_FLT[ch]}: {e}')
                logger.warning(f'SKIPPING template FLT evaluation for {POL_TEMPLATE_FLT[ch]}!')


    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


    try:
        logger.info(f'Performing template FLT for {bkg_data["traces"].shape[0]} background traces...')
        res_bkg = do_template_FLT(bkg_data,template_FLT_dict,CH_TEMPLATE_FLT)

        logger.info(f'Saving background results in {OUT_FILE_BKG}')
        np.savez(OUT_FILE_BKG,
                corr_max=res_bkg['corr_max'],
                idx_template_best=res_bkg['idx_template_best'],
                time_best=res_bkg['time_best'],
                snr=bkg_data['snr'])
    except:
        pass

    try:
        logger.info(f'Performing template FLT for {sig_data["traces"].shape[0]} signal traces...')
        res_sig = do_template_FLT(sig_data,template_FLT_dict,CH_TEMPLATE_FLT)

        logger.info(f'Saving signal results in {OUT_FILE_SIG}')
        np.savez(OUT_FILE_SIG,
                corr_max=res_sig['corr_max'],
                idx_template_best=res_sig['idx_template_best'],
                time_best=res_sig['time_best'],
                snr=sig_data['snr'])
    except:
        pass

    logger.info('*** END OF SCRIPT ***')