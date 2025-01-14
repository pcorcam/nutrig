#! /usr/bin/env python3

# system
import sys
import time
import argparse

# scipy
import numpy as np

# grandlib
import grand.dataio.root_trees as rt

# analysis tools
sys.path.insert(0, "/pbs/home/p/pcorrea/grand/nutrig/scripts/")
from nutrig.simu_analysis.tools import *


'''
Author: Pablo Correa
Date  : 5 January 2024

Description:
    TODO

Usage:
    python3 get_sim_params.py <sim_dir> -o <output_dir> -p <primary> -s <start> -e <end>
    
    Parser arguments are described below.
'''

# Parser arguments
def manage_args():

    parser = argparse.ArgumentParser(description="Convert electric fields to voltages.")

    parser.add_argument('input_dir',
                        type=str,
                        help='Directory containing two subdirectories:\
                              - `efield` : GRANDroot files of ZHaiRES/CoREAS simulations\
                              - `voltage`: GRANDroot files of corrsesponding voltage files')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Directory for the output files in npz format.')
    
    parser.add_argument('-p',
                        '--primary',
                        type=str,
                        default='Proton',
                        help='Primary cosmic-ray type for the GRANDroot files to analyze.')
    
    parser.add_argument('-s',
                        '--start',
                        type=int,
                        default=None,
                        help='First file in directories to get pulse shape parameters for.')
    
    parser.add_argument('-e',
                        '--end',
                        type=int,
                        default=None,
                        help='Last file in directories to get pulse shape parameters for.')
    
    return parser.parse_args()


# Main program
if __name__ == '__main__':
    # Set a timer
    t0 = time.time()
    print('Starting...')

    # Load parser arguments
    args       = manage_args()
    input_dir  = args.input_dir
    output_dir = args.output_dir
    primary    = args.primary
    start      = args.start
    end        = args.end

    # Makes sure to load efield files with a corresponding voltage file
    print('Loading efield & voltage simulation files in {}...'.format(input_dir))
    efield_files, voltage_files = get_sim_root_files(input_dir,substr=primary)

    t = time.time() - t0
    print('>>> Files loaded, time elapsed = {:.2f} seconds\n'.format(t))

    # Shower properties that we want to keep:
    # - energy
    # - zenith
    # - azimuth
    # - Xmax
    # - number of simulated DUs
    energy   = np.empty(len(efield_files[start:end]),dtype=float)
    zenith   = np.empty(len(efield_files[start:end]),dtype=float)
    azimuth  = np.empty(len(efield_files[start:end]),dtype=float)
    Xmax     = np.empty(len(efield_files[start:end]),dtype=float)
    n_du_sim = np.empty(len(efield_files[start:end]),dtype=float)
    
    print('Obtaining simulation parameters...')
    # Loop over all simulation files
    for i, efield_file in enumerate(efield_files[start:end]):
        # Load in relevant TTrees
        tshower = rt.TShower(efield_file)
        trun    = rt.TRun(efield_file)

        # Only one entry per file = 1 shower simulation per file
        tshower.get_entry(0)
        trun.get_entry(0)

        # Get shower parameters
        energy[i]   = tshower.energy_primary # [GeV]
        zenith[i]   = np.deg2rad(tshower.zenith) # [rad]
        azimuth[i]  = np.deg2rad(tshower.azimuth) # [rad]
        Xmax[i]     = tshower.xmax_grams # [g cm^-2]
        n_du_sim[i] = len(trun.du_id)

        # Close the associated root files to reduce memory usage
        # NOTE: this does not appear to work
        tshower.close_file()

        # Print progress message
        if not i%500:
            msg = '{}/{} files processed...'.format(i,len(efield_files[start:end]))
            print(msg)

    print('Saving...')
    npz_file = output_dir + 'sim_params_{}_{}.npz'.format(input_dir.split('/')[-2],primary.lower())
    np.savez(npz_file,
             energy=energy,
             zenith=zenith,
             azimuth=azimuth,
             Xmax=Xmax,
             n_du_sim=n_du_sim)
    
    t = time.time() - t0
    msg  = '>>> Done!'
    msg += '\n>>> Pulse parameters saved at {}'.format(npz_file)
    msg += '\n>>> Total time elapsed = {:.2f} seconds'.format(t)
    print(msg)
