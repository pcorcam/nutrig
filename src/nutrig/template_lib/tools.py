#! /usr/bin/env python3
'''
Tools for scripts in $NUTRIG/template_lib/v1/
'''

# system
import glob
import warnings

# scipy
import numpy as np
from scipy.signal import find_peaks

# grandlib
import grand.dataio.root_trees as rt # type: ignore


'''
© Pablo Correa, 5 January 2024

Description:
    This script contains the tools to:
    - analyze the pulse shapes of air-shower simulations (for GP300)
    - TODO: create a library of templates for a trigger algorithm
'''


def get_sim_root_files(sim_dir,
                       rf_chain,
                       substr=None):
    '''
    Description
    -----------
    Function that obtains the GRANDroot simulation files in `sim_dir`.
    Two lists are returned:
    - list of electric-field ZHaiRES/CoREAS simulation files, including shower properties;
    - list of corresponding voltage simulation files, see `convert_efield_to_voltage.py`.
    

    Arguments
    ---------
    `sim_dir`
    type        : str
    description : Directory where the simulations are stored in GRANDroot format.
                  Should contain `efield` and `voltage` subdirectories.

    `rf_chain`'
    type        : str
    description : Version of the RF chain used for the voltage simulations.
                  Example: rfv1, rfv1, ...

    `substr` (optional)
    type        : str
    description : Required substring in filenames.
                  Can be used to e.g. only select simulations for 'Proton'.

                                
    Returns
    -------
    `efield_files`
    type        : list[str]
    description : List containing all electric-field simulation files.

    `voltage_files`
    type        : list[str]
    description : List containing all corresponding voltage simulation files.
    '''

    efield_dir  = sim_dir + 'efield/'
    voltage_dir = sim_dir + f'voltage_{rf_chain}/'  

    efield_files  = np.array( sorted( glob.glob(efield_dir+'*.root') ) )
    voltage_files = np.array( sorted( glob.glob(voltage_dir+'*.root') ) )

    # Sometimes a bug occured in converting efield to voltage
    # Remove those efield files from the list
    # Add index j to speed up the process, arrays are sorted
    mask = np.ones(len(efield_files),dtype=bool)
    j = 0

    for i, efield_file in enumerate(efield_files):
        voltage_file = efield_file.replace('efield',f'voltage_{rf_chain}').replace('gr_','gr_voltage_')
        if voltage_file in voltage_files[i-j:]:
            continue
        else:
            mask[i] = False
            j += 1 
    #e_not = efield_files[np.logical_not(mask)]
    efield_files = efield_files[mask]

    # Only keep files with given substring
    if substr is not None:
        mask = [substr in efield_file for efield_file in efield_files]
        efield_files  = efield_files[mask]
        voltage_files = voltage_files[mask]

    # Need to convert back to list so that we can open the files with grandlib
    efield_files  = efield_files.tolist()
    voltage_files = voltage_files.tolist()

    return efield_files, voltage_files#, e_not


def digitize(trace,
             simu_sampling_rate=2e3,
             adc_sampling_rate=500,
             adc_to_voltage=0.9e6/2**13,
             quantize=True):
    '''
    Description
    -----------
    Performs the virtual digitization of voltage traces at the ADC level:
    - desamples the simulated signal to the ADC sampling rate;
    - converts voltage to ADC counts.
    NOTE: this step should already be included in a future grandlib version!

    Arguments
    ---------
    `trace`
    type        : np.ndarray[float]
    units       : µV
    description : Array of voltage traces at the ADC level with dimensions (3,N_simu_samples).

    `simu_sampling_rate`
    type        : float
    units       : MHz
    description : Sampling rate used in the ZHaiRES/CoREAS simulation. Typically 2 GHz.

    `adc_sampling_rate`
    type        : float
    units       : MHz
    description : Sampling rate of the ADC. Currently 500 MHz.

    `adc_to_voltage`
    type        : float
    units       : µV
    description : Conversion factor from ADC counts to voltage.

    `quantize`
    type        : bool
    description : Option to quantize the ADC signal as integers.
                  Default should be True. False is useful for tests / template construction.
    
                                
    Returns
    -------
    `trace`
    type        : np.ndarray[float]
    units       : LSB
    description : The digitized array of voltage traces, with the ADC sampling rate and in ADC counts.
    '''
    
    # Convert voltage to ADC
    trace = trace/adc_to_voltage
    
    # Truncate to get the closest integer
    if quantize:
        trace = np.trunc(trace)

    # Obtain desampling factor
    desampling_factor = int(simu_sampling_rate/adc_sampling_rate)

    if desampling_factor < 1:
        raise Exception('Simulation sampling rate can not be lower than ADC sampling rate!',desampling_factor)
    if not ( (desampling_factor & (desampling_factor-1) == 0) and desampling_factor != 0 ):
        warn = 'Desampling factor is not a power of 2: {}'.format(desampling_factor)
        warnings.warn(warn)

    # Return trace in ADC counts and with the ADC sampling rate
    return trace[:,::desampling_factor]


def above_thresh(trace,
                 thresh=15*5):
    '''
    Description
    -----------
    Checks if any of the abs(trace) values are above the desired threshold.

    Arguments
    ---------
    `trace`
    type        : np.ndarray[float]
    units       : LSB
    description : Array of ADC traces with dimensions (3,N_samples).

    `thresh`
    type        : float
    units       : LSB
    description : Threshold value. 
                  Default set to 5x average galactic noise threshold (~15 ADC counts).

    Returns
    -------
    type        : bool
    description : Whether the threshold is exceeded. 
    '''
    return np.any( np.abs(trace) > thresh )


# Get the largest peak-to-peak value of the trace
def get_peak_to_peak(trace):
    return np.max(trace,axis=1) - np.min(trace,axis=1)


# Obtain parameters that relate to the pulse shape:
# - the peak-to-peak voltage
# - the actual width of the pulse
# - the number of peaks
# - distance between the two largest peaks (abs value)
# - ratio between two largest peaks (abs value)
def get_pulse_params(trace,
                     thresh=15*5):
    
    # Compute for all directions (X,Y,Z)
    p2p        = get_peak_to_peak(trace)
    n_peaks    = np.empty(trace.shape[0])
    width      = np.empty(trace.shape[0])
    peak_ratio = np.empty(trace.shape[0])
    peak_dist  = np.empty(trace.shape[0])

    for i in range(len(width)):
        # Obtain the peak positions
        peak_positions, properties = find_peaks(np.abs(trace[i,:]),height=thresh)
        peak_heights = properties['peak_heights']
        n_peaks[i]   = len(peak_positions)

        # In case there are no peaks, tag it with -1
        if n_peaks[i] == 0:
            warn = 'No pulse above threshold!'
            warnings.warn(warn)
            width[i] = -1
            peak_ratio[i] = -1
            peak_dist[i] = -1
        # In case there is one peak, tag it with 0
        elif n_peaks[i] == 1:
            warn = 'Only one pulse above threshold!'
            warnings.warn(warn)
            width[i] = 0
            peak_ratio[i] = 0
            peak_dist[i] = 0
        else:
            # The width of the pulse is the position(last peak) - position(first peak)
            width[i] = peak_positions[-1] - peak_positions[0]
            
            # Sort maxima, find ratio and positions of the 2 largest ones
            sort_order = np.argsort(peak_heights)
            peak_heights = peak_heights[sort_order]
            peak_positions = peak_positions[sort_order]
            peak_ratio[i] = peak_heights[-1]/peak_heights[-2]
            peak_dist[i] = np.abs(peak_positions[-1] - peak_positions[-2])

    # Return the distance between the peaks
    return p2p, n_peaks, width, peak_ratio, peak_dist


# Compute the height at Xmax given a slant depth model of the atmopshere
def get_altitude_at_Xmax(Xmax,
                         zenith,
                         atmos_altitude,
                         atmos_depth):
    # Difference between "vertical equivalent Xmax" and array of slant depths
    # Assumes flat earth :)
    diff = np.abs( Xmax*np.cos( zenith ) - atmos_depth )

    # Find at which index the difference is the smallest
    idx = np.argmin(diff)

    # Estimate the height using the corresponding index
    return atmos_altitude[idx]


# Compute the opening angle between the shower direction and DU-Xmax direction
def get_omega(du_xyz,
              Xmax,
              zenith,
              azimuth,
              atmos_altitude,
              atmos_depth):
    
    # Get height corresponding to Xmax; assume flat atmosphere
    height_Xmax = get_altitude_at_Xmax(Xmax,zenith,atmos_altitude,atmos_depth)
    
    # Coordinate system: shower core at the origin, X = northing, Y = easting, Z = height
    # NOTE: ZHaireS simulations result in XY coordinates of DUs wrt to shower at the XY origin, 
    #       but where Z is the actual height (not w.r.t. the shower core)
    # Correct for this by substracting the average DU height (GP300 is between 1200-1300 m asl)
    # Not perfect but negligible, tens of meters of error compared to km scales
    if np.any(du_xyz[:,2]>1000):
        mean_du_height = np.mean(du_xyz[:,2])
        du_xyz[:,2] -= mean_du_height
        height_Xmax -= mean_du_height

    # Get the coordinates of Xmax
    Xmax_pos = np.array([height_Xmax*np.tan(zenith)*np.cos(azimuth),
                         height_Xmax*np.tan(zenith)*np.sin(azimuth),
                         height_Xmax])

    # Direction vector of shower
    k = - np.array([np.cos(azimuth)*np.sin(zenith),
                    np.sin(azimuth)*np.sin(zenith),
                    np.cos(zenith)])

    # Direction vectors of DUs pointing to Xmax
    dist_du_Xmax = np.linalg.norm(du_xyz-Xmax_pos,axis=1)
    dX = ( (du_xyz - Xmax_pos).T / dist_du_Xmax ).T

    # Get the cone opening angle (between shower axis and DU-Xmax)
    cos_omega = np.dot(dX,k)

    # Patch for numerical errors close to 1
    cos_omega[cos_omega>1.] = 1.
    
    return np.arccos(cos_omega)


def get_refraction_index_at_pos(X):
    '''
    Code by Marion Guelfand
    Based on ZHAireS parameters
    Shower coordinates (origin = core), units: m
    '''

    R_earth = 6370949 # [m]
    ns = 325
    kr = -0.1218

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+R_earth)**2 + R2 ) - R_earth)/1e3 # Altitude in km
    rh = ns*np.exp(kr*h)
    n = 1.+1e-6*rh
    #n = 1.+(1e-6*rh)/2
    return n


def get_omega_c(refraction_idx):
    return np.arccos(1/refraction_idx)


# Compute the opening angle between the shower direction and DU-Xmax direction
def get_omega_from_Xmax_pos(du_xyz,
                            Xmax_pos_shc):
    
    # Coordinate system: shower core at the origin, X = northing, Y = easting, Z = height
    # NOTE: ZHaireS simulations result in XY coordinates of DUs wrt to shower at the XY origin, 
    #       but where Z is the actual height (not w.r.t. the shower core)
    # Correct for this by substracting the average DU height (GP300 is between 1200-1300 m asl)
    # Not perfect but negligible, tens of meters of error compared to km scales
    if np.any(du_xyz[:,2]>1000):
        mean_du_height = np.mean(du_xyz[:,2])
        du_xyz[:,2] -= mean_du_height

    # Direction vector of shower
    k = - Xmax_pos_shc / np.linalg.norm(Xmax_pos_shc)

    # Direction vectors of DUs pointing to Xmax
    dist_du_Xmax = np.linalg.norm(du_xyz-Xmax_pos_shc,axis=1)
    dX = ( (du_xyz - Xmax_pos_shc).T / dist_du_Xmax ).T

    # Get the cone opening angle (between shower axis and DU-Xmax)
    cos_omega = np.dot(dX,k)

    # Patch for numerical errors close to 1
    cos_omega[cos_omega>1.] = 1.
    
    return np.arccos(cos_omega)


def set_trace_length(trace,
                     target_length):
    
    in_trace_length = trace.shape[-1]
    
    # Pad with zeros at end if trace length is smaller target length
    if in_trace_length <= target_length:
        out_trace = np.zeros( (trace.shape[0],target_length) )
        out_trace[:,:in_trace_length] = trace

    # Pulse simulations are normally at the beginning of trace
    else:
        out_trace = trace[:,:target_length]
    
    return out_trace


def load_traces(trace_dir,
                trace_length=1024):

    trace_files = sorted( glob.glob(trace_dir+'/*.npz') )
    traces      = np.zeros( (1,3,trace_length) )

    for trace_file in trace_files[10:11]:
        f      = np.load(trace_file)
        traces = np.concatenate( (traces,f['traces_adc']) )

    return traces[1:]