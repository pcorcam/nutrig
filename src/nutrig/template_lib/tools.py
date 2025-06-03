#! /usr/bin/env python3
'''
Tools for scripts in $NUTRIG/template_lib/v2/
'''

# system
import sys
import warnings
import logging

# scipy
import numpy as np
from scipy.signal import find_peaks

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
                            shower_core_pos,
                            Xmax_pos_shc):
    
    # Coordinate system: shower core at the origin, X = northing, Y = easting, Z = height
    # NOTE: ZHaireS simulations result in XY coordinates of DUs wrt to shower at the XY origin, 
    #       but where Z is the actual height (not w.r.t. the shower core)
    # Correct for this by substracting the average DU height (GP300 is between 1200-1300 m asl)
    # Not perfect but negligible, tens of meters of error compared to km scales
    if np.any(du_xyz[:,2]>1000):
        mean_du_height = np.mean(du_xyz[:,2])
        du_xyz[:,2] -= mean_du_height

    # Transform du_xyz to shower-core frame
    du_xyz_shc = du_xyz - shower_core_pos

    # Direction vector of shower
    k = - Xmax_pos_shc / np.linalg.norm(Xmax_pos_shc)

    # Direction vectors of DUs pointing to Xmax
    dist_du_Xmax = np.linalg.norm(du_xyz_shc-Xmax_pos_shc,axis=1)
    dX = ( (du_xyz_shc - Xmax_pos_shc).T / dist_du_Xmax ).T

    # Get the cone opening angle (between shower axis and DU-Xmax)
    cos_omega = np.dot(dX,k)

    # Patch for numerical errors close to 1
    cos_omega[cos_omega>1.] = 1.
    
    return np.arccos(cos_omega)