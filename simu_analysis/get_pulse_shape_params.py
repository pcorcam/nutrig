#! /usr/bin/env python3

# system
import time
import argparse

# scipy
import numpy as np

# grandlib
import grand.dataio.root_trees as rt

# analysis tools
from nutrig.simu_analysis.tools import *

'''
Author: Pablo Correa
Date  : 5 January 2024

Description:
    TODO

Usage:
    python3 get_pulse_shape_params.py <sim_dir> -o <output_dir> -p <primary> -s <start> -e <end>
    
    Parser arguments are described below.
'''

# Parser arguments
def manage_args():

    parser = argparse.ArgumentParser(description="Convert electric fields to voltages.")

    parser.add_argument('input_dir',
                        type=str,
                        help='Directory containing two subdirectories:\
                              - `efield` : GRANDroot files of ZHaiRES/CoREAS simulations\
                              - `voltage_{rf_chain}`: GRANDroot files of corrsesponding voltage files')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Directory for the output files in npz format.')
    
    parser.add_argument('-p',
                        '--primary',
                        type=str,
                        default='Proton',
                        help='Primary cosmic-ray type for the GRANDroot files to analyze.')
    
    parser.add_argument('-rf',
                        '--rf_chain',
                        type=str,
                        default='rfv2',
                        help='Version of RF chain used for voltage simulations.')
    
    parser.add_argument('-t',
                        '--thresh',
                        type=int,
                        default=75,
                        help='ADC threshold required to select a trace.')
    
    parser.add_argument('-s',
                        '--start',
                        type=int,
                        default=0,
                        help='First file in directories to get pulse shape parameters for.')
    
    parser.add_argument('-e',
                        '--end',
                        type=int,
                        default=500,
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
    rf_chain   = args.rf_chain
    thresh     = args.thresh
    start      = args.start
    end        = args.end

    print('Loading efield & voltage simulation files in {}...'.format(input_dir))
    efield_files, voltage_files = get_sim_root_files(input_dir,rf_chain,substr=primary)

    t = time.time() - t0
    print('>>> Files loaded, time elapsed = {:.2f} seconds\n'.format(t))

    # Shower properties that we want to keep:
    # - energy
    # - zenith
    # - azimuth
    # - opening angle omega
    # - Cherenkov angle
    energy       = np.empty(1,dtype=float)
    zenith       = np.empty(1,dtype=float)
    azimuth      = np.empty(1,dtype=float)
    omega        = np.empty(1,dtype=float)
    omega_c      = np.empty(1,dtype=float)

    # Trace properties that we are interested in for pulses that exceed galactic noise threshold:
    # - the peak-to-peak voltage
    # - the actual width of the pulse
    # - the number of peaks
    # - distance between the two largest peaks
    peak_to_peak = np.empty((1,3),dtype=float)
    n_peaks      = np.empty((1,3),dtype=float)
    pulse_width  = np.empty((1,3),dtype=float)
    peak_ratio   = np.empty((1,3),dtype=float)
    peak_dist    = np.empty((1,3),dtype=float)

    # Other useful stuff to keep for bookkeeping purposes
    du_idx       = np.empty(1,dtype=int)
    vfile        = np.empty(1,dtype=str)

    print('Obtaining pulse shape parameters...')
    # Loop over all simulation files
    for voltage_file, efield_file in zip(voltage_files[start:end],efield_files[start:end]):
        print('For',voltage_file)
        # Load in relevant TTrees
        tvoltage    = rt.TVoltage(voltage_file)
        tshower     = rt.TShower(efield_file)
        tshowersim  = rt.TShowerSim(efield_file)
        trun        = rt.TRun(efield_file)

        # Only one entry per file = 1 shower simulation per file
        trun.get_entry(0); tvoltage.get_entry(0); tshower.get_entry(0) #; tshowersim.get_entry(0)

        # Positions of DUs where signal is simulated
        # Given in coordinate system where shower core position = origin 
        # Effectively du_xyz_gp300 - shower_core_pos
        du_xyz           = np.array(trun.du_xyz) # [m]
        du_count         = tvoltage.du_count

        # Shower parameters
        Xmax_pos_shc     = tshower.xmax_pos_shc # [m]
        n                = get_refraction_index_at_pos(Xmax_pos_shc)
        # to save
        zenith_new       = np.deg2rad(tshower.zenith) # [rad]
        azimuth_new      = np.deg2rad(tshower.azimuth) # [rad]
        energy_new       = tshower.energy_primary # [GeV]
        primary_type_new = tshower.primary_type
        omega_new        = np.empty(du_count)
        omega_c_new      = get_omega_c(n) # [rad]

        # Atmosphere parameters
        #atmos_altitude   = np.array( tshowersim.atmos_altitude[0] ) # [m]
        #atmos_depth      = np.array( tshowersim.atmos_depth[0] ) # [g cm^-2]

        # Trace propetries that we want to save
        peak_to_peak_new = np.empty((du_count,3))
        n_peaks_new      = np.empty((du_count,3))
        pulse_width_new  = np.empty((du_count,3))
        peak_ratio_new   = np.empty((du_count,3))
        peak_dist_new    = np.empty((du_count,3))

        vfile_new        = np.empty(du_count,dtype='<U60')
        vfile_new.fill(voltage_file.split('/')[-1])
        
        # Mask to only save info for interesting traces
        mask = np.ones(du_count,dtype=bool)

        # Loop over all DUs
        for du in range(du_count):
            # Digitize the traces: go from voltage to ADC counts with the ADC's sampling rate
            trace = digitize( np.array(tvoltage.trace[du]) )

            # Only keep data of those DUs where noise threshold (5x galactic noise) is exceeded
            if not above_thresh(trace,thresh=thresh):
                mask[du] = False
                continue

            pulse_params = get_pulse_params(trace,thresh=thresh)

            peak_to_peak_new[du] = pulse_params[0]
            n_peaks_new[du]      = pulse_params[1]
            pulse_width_new[du]  = pulse_params[2]
            peak_ratio_new[du]   = pulse_params[3]
            peak_dist_new[du]    = pulse_params[4]

        # If there is nothing interesting, skip
        if not np.any(mask):
            continue
            
        # Compute omega
        #omega_new = get_omega(du_xyz[mask],Xmax,zenith_new,azimuth_new,atmos_altitude,atmos_depth)
        omega_new = get_omega_from_Xmax_pos(du_xyz[mask],Xmax_pos_shc)
        
        # For repeating shower parameter values
        ones = np.ones(len(omega_new))

        # Save interesting stuff
        energy  = np.hstack( (energy,energy_new*ones) )
        zenith  = np.hstack( (zenith,np.rad2deg(zenith_new)*ones) )
        azimuth = np.hstack( (azimuth,np.rad2deg(azimuth_new)*ones) )
        omega   = np.hstack( (omega,np.rad2deg(omega_new)) )
        omega_c = np.hstack( (omega_c,np.rad2deg(omega_c_new)*ones) )

        peak_to_peak = np.vstack( (peak_to_peak,peak_to_peak_new[mask]) )
        n_peaks      = np.vstack( (n_peaks,n_peaks_new[mask]) )
        pulse_width  = np.vstack( (pulse_width,pulse_width_new[mask]) )
        peak_ratio   = np.vstack( (peak_ratio,peak_ratio_new[mask]))
        peak_dist    = np.vstack( (peak_dist,peak_dist_new[mask]) )

        du_idx = np.hstack( (du_idx,np.arange(du_count)[mask]) )
        vfile  = np.hstack( (vfile,vfile_new[mask]) )

        # Close the associated root files to reduce memory usage
        # NOTE: this does not appear to work
        trun.stop_using()
        tvoltage.stop_using()
        tshower.stop_using()

        trun.close_file()
        tvoltage.close_file()
        tshower.close_file()

    print('Saving...')
    npz_file = output_dir + 'pulse_shape_params_{}_{}_{}_thresh_{}_files_{}_{}.npz'.format(input_dir.split('/')[-4],primary.lower(),rf_chain,thresh,start,end)
    np.savez(npz_file,
             energy=energy[1:],
             zenith=zenith[1:],
             azimuth=azimuth[1:],
             omega=omega[1:],
             omega_c=omega_c[1:],
             peak_to_peak=peak_to_peak[1:],
             n_peaks=n_peaks[1:],
             pulse_width=pulse_width[1:],
             peak_ratio=peak_ratio[1:],
             peak_dist=peak_dist[1:],
             du_idx=du_idx[1:],
             vfile=vfile[1:])
    
    t = time.time() - t0
    msg  = '>>> Done!'
    msg += '\n>>> Pulse parameters saved at {}'.format(npz_file)
    msg += '\n>>> Total time elapsed = {:.2f} seconds'.format(t)
    print(msg)

