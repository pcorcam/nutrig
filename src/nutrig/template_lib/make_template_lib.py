'''
This scripts creates a library of templates for the NUTRIG template-fit method.
'''

###########
# IMPORTS #
###########

import os
import glob
import logging
import argparse

import numpy as np

import grand.dataio.root_trees as rt

import template_lib.tools as tools

from database.tools import filter_traces_bandpass

import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.style.use('/pbs/home/p/pcorrea/tools/matplotlib_style_sans-serif.txt')

logger = logging.getLogger(__name__)


####################
# FIXED PARAMETERS #
####################

SIM_SOFTWARE = 'ZHAireS'
SIM_TAG      = 'DC2.1rc2'
THRESH       = 100 # ADC counts
POL          = 'XY'
FILTER_TAG   = 'FILTERED'

PRESEL_BASEDIR = '/sps/grand/pcorrea/nutrig/template/v2/preselection'
PRESEL_DIR     = os.path.join(PRESEL_BASEDIR,f'{SIM_SOFTWARE}_{SIM_TAG}_{FILTER_TAG}_{POL}_THRESH_{THRESH}')
N_SAMPLES      = 100

N_TEMPLATES     = 5
OMEGA_RATIO_MIN = 0
OMEGA_RATIO_MAX = 2
OMEGA_BINS      = np.linspace(OMEGA_RATIO_MIN,OMEGA_RATIO_MAX,N_TEMPLATES+1)

OUT_DIR      = PRESEL_BASEDIR.replace('preselection','lib')
OUT_FILE     = os.path.join( OUT_DIR,f'templates_{SIM_SOFTWARE}_{SIM_TAG}_{FILTER_TAG}_{POL}_THRESH_{THRESH}_{N_TEMPLATES}.npz' )

PLOT_DIR      = '/pbs/home/p/pcorrea/grand/nutrig/template_lib/v2/plots'
PLOT_FILENAME = f'corr_hist_{FILTER_TAG}_{POL}_THRESH_{THRESH}_{N_TEMPLATES}'
PLOT_TITLE    = f'Template selection {FILTER_TAG}, POL = {POL}, THRESH = {THRESH}'
PLOT_SUPTITLE = f'{SIM_SOFTWARE} {SIM_TAG}'

#############
# FUNCTIONS #
#############

def load_templates_presel_in_bin(presel_dir,
                                 omega_bin):

    presel_files = sorted( glob.glob( os.path.join(presel_dir,'*.npz') ) )

    templates   = np.zeros((0,100))
    omega_ratio = np.zeros((0,))

    for file in presel_files:
        with np.load(file) as f:
            templates   = np.concatenate((templates,f['templates']))
            omega_ratio = np.concatenate((omega_ratio,f['omega']/f['omega_c']))

    mask_omega_bin = np.logical_and( np.where( omega_ratio >= omega_bin[0],True,False ), np.where( omega_ratio < omega_bin[1],True,False ) )

    templates_bin = templates[mask_omega_bin]

    return templates_bin


def compute_cross_corr_grid_in_bin(templates_presel_in_bin):

    n_templates = len(templates_presel_in_bin)
    corr_grid   = np.zeros((n_templates,n_templates))

    for i in range(n_templates):
        for j in range(n_templates):
            corr_grid[i,j] = np.correlate(templates_presel_in_bin[i],templates_presel_in_bin[j],mode='valid')[0]

    corr_grid = np.clip( np.abs(corr_grid),0.,1. ) # fix rounding errors

    return corr_grid


def find_best_template_in_bin(templates_presel_in_bin,
                              corr_grid_in_bin):

    corr_mean = np.mean(corr_grid_in_bin,axis=1)
    idx_best  = np.argmax(corr_mean)

    return templates_presel_in_bin[idx_best]


def plot_corr_hist_in_bin(corr_grid_in_bin,
                          omega_bin):

    # rows, cols       = np.triu_indices(corr_grid_in_bin.shape[0],k=1)
    # corr_vals_unique = corr_grid_in_bin[rows,cols]

    corr_mean = np.mean(corr_grid_in_bin,axis=1)

    plot_title = PLOT_TITLE + rf', $\omega/\omega_c \in [{omega_bin[0]:.1f},{omega_bin[1]:.1f}]$'

    fig, ax = plt.subplots()

    ax.hist(corr_mean,bins=np.linspace(0,1,11))
    ax.text(0.1,0.9,rf'$\max \langle \rho \rangle = {np.max(corr_mean):.2f}$',transform=plt.gca().transAxes)

    ax.set_yscale('log')

    ax.set_xlabel(r'$\langle \rho \rangle$')
    ax.set_ylabel('Counts')

    ax.set_title(plot_title,fontsize=15)
    plt.suptitle(PLOT_SUPTITLE)

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    plot_file = os.path.join(PLOT_DIR,PLOT_FILENAME+f'_{omega_bin[0]:.1f}_{omega_bin[1]:.1f}.png')
    plt.savefig(plot_file,dpi=200)

    logger.info(f'Saved correlation histogram in {plot_file}')

    return


###########################
# DEFINE PARSER ARGUMENTS #
###########################

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Find background pulses in a GP80 MD data file.")

    parser.add_argument('-b',
                        dest='bin',
                        default=0,
                        type=int,
                        help='Specify the bin for which to compute the correlation values.')

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
    logger.info(f'*** Making template library from preselected templates in {PRESEL_DIR} ***')
    logger.info(f'*** Selecting {N_TEMPLATES} templates from the following omega/omega_c bins: {OMEGA_BINS} ***')

    templates_final = np.zeros( (N_TEMPLATES,N_SAMPLES) )

    for i in range(N_TEMPLATES):
        omega_bin = OMEGA_BINS[i:i+2]
        logger.info(f'Finding best template in bin {i+1}/{N_TEMPLATES}: omega/omega_c in {omega_bin}')

        templates_presel_bin = load_templates_presel_in_bin(PRESEL_DIR,omega_bin)
        corr_grid            = compute_cross_corr_grid_in_bin(templates_presel_bin)
        template_best        = find_best_template_in_bin(templates_presel_bin,corr_grid)

        plot_corr_hist_in_bin(corr_grid,omega_bin)
        
        templates_final[i] = template_best

    # Create outdir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    logger.info(f'Saving templates in {OUT_FILE}')
    np.savez(OUT_FILE,templates=templates_final)

    logger.info(f'Saving templates in {OUT_FILE.replace("npz","txt")}')

    header  = '***TEMPLATE SELECTION FOR NUTRIG FLT STUDY***\n'
    header += f'Number of templates (=rows): {N_TEMPLATES}\n'
    header += f'Number of samples of 2 ns per template (=columns): {templates_final.shape[-1]}\n'
    header += f'Templates selected from: {SIM_SOFTWARE} {SIM_TAG} \n\n'
    np.savetxt(OUT_FILE.replace('npz','txt'),templates_final,fmt='%.6e',header=header)

    logger.info('*** END OF SCRIPT ***')