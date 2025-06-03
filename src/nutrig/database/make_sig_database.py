###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import logging
import argparse

import numpy as np

import database.tools as tools

logger = logging.getLogger(__name__)


'''

for each sim file -> select N_traces noise traces
get_noise_traces

add signal to noise -> inject at random positions weighted by time distributiuon of bkg pulses
add_signal_to_noise
    tools.get_inj_weights()

filter + trigger
tools.find_thresh_trig()

calculate SNR -> peak at inj time / RMS in part of trace without pulse ???
get_snr ?

save file with all traces if there was at least one trigger

per event:
- all traces
- trigger tag
- inj time tag
- SNR

'''

###-###-###-###-###-###-###- FIXED PARAMETERS -###-###-###-###-###-###-###

'''
Filters and thresholds are fixed based on analysis with
`test_pretrigger.py` and `find_pretrigger_threshold.ipynb`
'''

freq_highpass = 50 # [MHz]
freqs_notch   = [50.2,55.1,126] # [MHz]
bw_notch      = [1.,1.,25.] # [MHz]

threshold1              = 35#35#55 # [ADC counts]
threshold2              = 25#25#35 # [ADC counts]
samples_from_trace_edge = 100 # [ADC samples]
include_Z               = False


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description='Makes a database of air-shower traces with added measured noise for offline trigger studies.')

    parser.add_argument('input_file',
                        type=str,
                        help='Input file containing voltage traces in GrandRoot format.')
    
    parser.add_argument('-n',
                        '--noise_dir',
                        type=str,
                        default='/sps/grand/pcorrea/nutrig/database/bkg/gp13_pretrigger_stationary_th1_55_th2_35/',
                        help='Directory containing noise traces in npz format.')
    
    parser.add_argument('-bp',
                        '--bkg_pulse_file',
                        type=str,
                        default='/sps/grand/pcorrea/nutrig/database/bkg/bkg_dataset_nutrig_gp13_train_seed_300.npz',
                        help='File containing background pulses in npz format.')
    
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='./',
                        help='Directory where the processed air-shower simulations will be stored in npz format.')
    
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=300, # for GP300 :)
                        help='Seed for random number generator.')
    
    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')
    
    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    #-#-#- Get parser arguments -#-#-#
    args           = manage_args()
    input_file     = args.input_file
    noise_dir      = args.noise_dir
    bkg_pulse_file = args.bkg_pulse_file
    output_dir     = args.output_dir
    seed           = args.seed
    verbose        = args.verbose

    logger = tools.load_logger(verbose)
    logger.info('*** START OF SCRIPT ***')

    #-#-#- Set the random number generator -#-#-#
    rng = np.random.default_rng(seed)

    #-#-#- Get the simulated air-shower traces -#-#-#
    #sim_traces, shower_core_pos, du_xyz, du_seconds, du_nanoseconds = tools.get_sim_traces(input_file)
    shower_params = tools.extract_shower_params(input_file)


    #-#-#- Get the DU IDs from the DU coordinates -#-#-#
    du_ids = tools.get_du_ids( shower_params['du_xyz'] )

    #-#-#- Collect random noise traces from measured data -#-#-#
    noise_traces, noise_file, noise_file_entries = tools.get_noise_traces(noise_dir,shower_params['traces'].shape[0],rng=rng)
    
    #-#-#- Add air-shower pulses  -#-#-#
    inj_weights                 = tools.get_pulse_inj_weights(bkg_pulse_file)
    sig_traces, inj_pulse_times = tools.add_sim_to_noise(shower_params['traces'],
                                                         noise_traces,
                                                         inj_weights=inj_weights,
                                                         rng=rng) 

    #-#-#- Perform the threshold trigger on all FILTERED traces -#-#-#
    logger.info('Filtering signal traces...')
    sig_traces_filtered = tools.filter_traces(sig_traces,
                                              freq_highpass=freq_highpass,
                                              freqs_notch=freqs_notch,
                                              bw_notch=bw_notch)
    
    logger.info('Performing double-threshold pretrigger...')
    pretrig_flags, pretrig_times = tools.find_thresh_triggers(sig_traces_filtered,
                                                              threshold1=threshold1,
                                                              threshold2=threshold2,
                                                              samples_from_trace_edge=samples_from_trace_edge,
                                                              include_Z=include_Z)
    mask_trig = np.where(pretrig_flags==0,False,True)
    
    #-#-#- If there are any triggers, calculate the SNR and save npz file containing all traces & trigger info of the event -#-#-#
    if np.any(mask_trig):
        logger.info(f'Triggered on {len(mask_trig[mask_trig==True])}/{len(mask_trig)} DUs!')

        snr = tools.get_snr(sig_traces_filtered,inj_pulse_times)
        logger.debug(snr)
        logger.debug(snr[mask_trig])
        
        output_filename = os.path.basename(input_file)
        output_filename = output_filename.replace('gr_voltage','sim_plus_noise')
        output_filename = output_filename.replace('.root',f'_seed_{seed}.npz')

        output_file = os.path.join(output_dir,output_filename)
        logger.info(f'Saving event with sim+noise traces at {output_file}')

        np.savez(output_file,
                 traces=sig_traces_filtered,
                 snr=snr,
                 inj_pulse_times=inj_pulse_times,
                 pretrig_flags=pretrig_flags,
                 pretrig_times=pretrig_times,
                 shower_core_pos=shower_params['shower_core_pos'],
                 xmax_pos_shc=shower_params['xmax_pos_shc'],
                 du_ids=du_ids,
                 du_xyz=shower_params['du_xyz'],
                 du_seconds=shower_params['du_seconds'],
                 du_nanoseconds=shower_params['du_nanoseconds'],
                 omega=shower_params['omega'],
                 omega_c=shower_params['omega_c'],
                 energy=shower_params['energy'],
                 zenith=shower_params['zenith'],
                 azimuth=shower_params['azimuth'],
                 noise_file=noise_file,
                 noise_file_entries=noise_file_entries)
        
    else:
        logger.info('No DUs were triggered!')

    logger.info('*** END OF SCRIPT ***')