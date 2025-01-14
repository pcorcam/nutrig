#! /usr/bin/env python3

import os
import glob
import argparse


'''
Pablo Correa, 13 December 2023
Updated on 4 April 2024

Description:
    Script to compute ADC voltages without galactic noise from air-shower electric fields.
    The goal is to create a library of ADC voltages for GRAND-trigger development (NUTRIG).
    The conversion is performed using the `convert_efield2voltage.py` script of GRANDlib.
    NOTE: with the developments of GRANDlib, this script might be outdated. Last use is with `dev_snonis`.

Usage:
    python3 convert_efield_to_voltage.py <input_dir> -o <output_dir> -s <start> -e <end>
    
    Parser arguments are described below.
'''

# Parser arguments
def manage_args():

    parser = argparse.ArgumentParser(description="Convert electric fields to voltages.")

    parser.add_argument('input_dir',
                        type=str,
                        help='Directory containing GRANDroot files with\
                              electric-field simulations of air showers.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Directory for the output files in GRANDroot\
                              format containing voltage traces.')
    
    parser.add_argument('-s',
                        '--start',
                        type=int,
                        default=0,
                        help='First file in directory to compute voltages.')
    
    parser.add_argument('-e',
                        '--end',
                        type=int,
                        default=None,
                        help='Last file in directory to compute voltages.')
    
    return parser.parse_args()


# Main program
if __name__ == "__main__":
    # Load parser arguments
    args       = manage_args()
    input_dir  = args.input_dir
    output_dir = args.output_dir
    start      = args.start
    end        = args.end

    # Get the input GRANDroot files containig electric field traces
    # Sorted so that it's easier to debug
    # Only take the files between specified limits
    input_files = sorted( glob.glob(input_dir+'*root') )
    if end is None:
        input_files = input_files[start:]
    else:
        input_files = input_files[start:end]

    # Take the input directory as output directory if none specified
    if output_dir == None:
        msg  = '\nWARNING: No output directory specified.'
        msg += '\nThe input directory will be used as output directory.'
        output_dir = input_dir
        print(msg)

    # Run the `convert_efield2voltage.py` script for the selected files
    # NOTE: for trigger purposes, we want no galactic noise, hence `--no_noise`
    for input_file in input_files:
        # Output directory -> voltage
        output_file = input_file.split('/')[-1].replace('gr_','gr_voltage_')
        cmd = 'python3 /pbs/home/p/pcorrea/grand/grandlib/scripts/convert_efield2voltage.py --no_noise --no_rf_chain --rf_chain_nut -o {} {}'
        cmd = cmd.format(output_dir+output_file,input_file)
        os.system(cmd)

'''
END
'''