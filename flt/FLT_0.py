#!/pbs/home/p/pcorrea/.conda/envs/grandlib2409/bin/python3.11
'''
Module for the offline FLT-0 (also referred to as the pretrigger or L1 trigger).

Adapted from https://github.com/watertien/T1_offline_trigger
CREDIT: Xishui Tian

NOTE: assumes an ADC sampling rate of 500 MHz -> 1 sample = 2 ns.
'''

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import logging
import sys

import numpy as np

import matplotlib.pyplot as plt

###-###-###-###-###-###-###- LOGGER SETUP -###-###-###-###-###-###-###

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s")
handler.setFormatter(formatter)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)


###-###-###-###-###-###-###- FLT-0 class -###-###-###-###-###-###-###

class FLT0:
    def __init__(self):
        '''
        Initialiser of the FLT0 class.
        '''

        # Parameters of the FLT-0. Mymics the settings of the online FLT-0.
        # Not all parameters are relevant for the offline FLT-0, but stored anyways for completeness.
        # NOTE: all times are in nanoseconds!
        self.trigger_config = dict([("t_quiet", 512), # [ns]; Quiet time before the first T1 crossing
                                    ("t_period", 512), # [ns]; Time after first T1 crossing where the trigger conditions are evaluated
                                    ("t_sepmax", 20), # [ns]; Maximum time between two T2 crossings
                                    ("nc_min", 2), # Minimum number of T2 crossings
                                    ("nc_max", 8), # Maximum number of T2 crossings
                                    ("q_min", 0), # Minimum charge (charge = pulse peak / number of T2 crossings)
                                    ("q_max", 255), # Maximum charge
                                    ("th1", 35), # T1 threshold
                                    ("th2", 25), # T2 threshold
                                    # Configs of readout timewindow
                                    ("t_pretrig", 960), # [ns]; Readout time of trace before the trigger decision (in the first triggered channel)
                                    ("t_overlap", 64), # [ns]; Overlap time for trigger decisions between different channels [ns]
                                    ("t_posttrig", 1024) # [ns]; Readout time after pretrigger+overlap times [ns]
                                    ])

        return
    

    def set_trigger_config(self,
                           trigger_config):
        '''
        Setter for the FLT-0 configuration parameters.

        Arguments
        ---------

        `trigger_config`
        type        : dict
        description : Configuration parameters for the FLT-0.
                      Does not need to include all keys; only those to be changed from default are sufficient.
        '''
        
        for key in trigger_config.keys():
            assert key in self.trigger_config.keys(), f'{key} is not a valid parameter of the FLT-0, must be in {self.trigger_params.keys()}'
            self.trigger_config[key] = trigger_config[key]

        return
    
    
    def _extract_trigger_parameters(self,
                                    trace,
                                    baseline=0):
        '''
        Extracts the relevant parameters of a trace for the trigger assessment.
        Mimics the online FLT-0 trigger on the GP300 firmware (v1.b).

        CREDIT: XISHUI TIAN. Adapted from https://github.com/grand-mother/GP13_offline_trigger 

        Arguments
        ---------
        `trace`
        type        : np.ndarray[float]
        units       : ADC counts
        description : ADC trace of one channel, with shape (N_samples,).

        `trigger_config`
        type        : dict
        description : Configuration parameters for the FLT-0.

        `baseline`
        type        : int
        description : Baseline of the trace used to calculate the charge parameter Q.
        
                                    
        Returns
        -------
        `dict_trigger_infos`
        type        : dict
        description : Parameters relevant to the FLT-0:
                      - `index_T1_crossing` : index of first T1 crossing
                      - `index_T2_crossing` : indices of all T2 crossings
                      - `NC` : number of T2 crossings (NC)
                      - `Q` : charge Q = peak/NC
        '''

        # Find the position of the first T1 crossing
        index_t1_crossing = np.where((trace) > self.trigger_config["th1"],
                                    np.arange(len(trace)), -1)
        dict_trigger_infos = dict()

        mask_T1_crossing = (index_t1_crossing != -1)
        if sum(mask_T1_crossing) == 0:
            # No T1 crossing 
            logger.error('No T1 crossing')
            raise ValueError("No T1 crossing!")
            
        dict_trigger_infos['index_T1_crossing'] = None
        # Tquiet to decide the quiet time before the T1 crossing 
        for i in index_t1_crossing[mask_T1_crossing]:
            # Abs value not exceeds the T1 threshold
            if i - self.trigger_config["t_quiet"]//2 < 0:
                logger.error('Not enough data before T1 crossing!')
                raise ValueError("Not enough data before T1 crossing!")
            if np.all((trace[np.max([0, i - self.trigger_config['t_quiet'] // 2]):i]) <= self.trigger_config["th1"]):
                dict_trigger_infos["index_T1_crossing"] = i
                # the first T1 crossing satisfying the quiet condition
                break
        if dict_trigger_infos['index_T1_crossing'] == None:
            logger.error('No T1 crossing with Tquiet satified!')
            plt.plot(trace)
            plt.scatter(index_t1_crossing[mask_T1_crossing],trace[mask_T1_crossing],color='r',s=20,marker='o')
            plt.axhline(self.trigger_config["th1"],color='k',ls='--')
            plt.axhline(self.trigger_config["th2"],color='k',ls=':')
            plt.xlim([np.argmax(trace)-150,np.argmax(trace)+150])
            plt.show()
            raise ValueError("No T1 crossing with Tquiet satified!")
        
        # The trigger logic works for the timewindow given by T_period after T1 crossing.
        # Count number of T2 crossings, relevant pars: T2, NCmin, NCmax, T_sepmax
        # From ns to index, divided by two for 500MHz sampling rate
        period_after_T1_crossing = trace[dict_trigger_infos["index_T1_crossing"]:dict_trigger_infos["index_T1_crossing"]+self.trigger_config['t_period']//2]
        
        # All the points above +T2
        positive_T2_crossing = (np.array(period_after_T1_crossing) > self.trigger_config['th2']).astype(int)
        
        # Positive crossing, the point before which is below T2.
        mask_T2_crossing_positive = np.diff(positive_T2_crossing) == 1
        # if np.sum(mask_T2_crossing_positive) > 0:
        #     index_T2_crossing_positive = np.arange(len(period_after_T1_crossing) - 1)[mask_T2_crossing_positive]
        # negative_T2_crossing = (np.array(period_after_T1_crossing) < - self.trigger_config['th2']).astype(int)
        # mask_T2_crossing_negative = np.diff(negative_T2_crossing) == 1
        # if np.sum(mask_T2_crossing_negative) > 0:
        #     index_T2_crossing_negative = np.arange(len(period_after_T1_crossing) - 1)[mask_T2_crossing_negative]
        # n_T2_crossing_negative = np.len(index_T2_crossing_positive)

        # Register the first T1 crossing as a T2 crossing
        mask_first_T1_crossing = np.zeros(len(period_after_T1_crossing), dtype=bool)
        mask_first_T1_crossing[0] = True
        # mask_first_T1_crossing[1:] = (mask_T2_crossing_positive | mask_T2_crossing_negative)
        mask_first_T1_crossing[1:] = (mask_T2_crossing_positive)
        index_T2_crossing = np.arange(len(period_after_T1_crossing))[mask_first_T1_crossing]
        
        n_T2_crossing = 1 # Starting from the first T1 crossing.
        dict_trigger_infos["index_T2_crossing"] = [0]
        if len(index_T2_crossing) > 1:
            for i, j in zip(index_T2_crossing[:-1], index_T2_crossing[1:]):
                # The separation between successive T2 crossings
                time_separation = (j - i) * 2
                if time_separation < self.trigger_config["t_sepmax"]:
                    n_T2_crossing += 1
                    dict_trigger_infos["index_T2_crossing"].append(j)
                else:
                    # Violate the maximum separation, fail to trigger
                    logger.error(f"Violating Tsepmax, the separation is {time_separation} ns.")
                    plt.plot(trace)
                    plt.scatter(index_t1_crossing[mask_T1_crossing],trace[mask_T1_crossing],color='r',s=20,marker='o')
                    plt.axhline(self.trigger_config["th1"],color='k',ls='--')
                    plt.axhline(self.trigger_config["th2"],color='k',ls=':')
                    plt.xlim([np.argmax(trace)-150,np.argmax(trace)+300])
                    plt.show()
                    raise ValueError(f"Violating Tsepmax, the separation is {time_separation} ns.")
        else:
            n_T2_crossing = 1
            j = 1

        # Change the reference of indices of T2 crossing
        dict_trigger_infos["index_T2_crossing"] = np.array(dict_trigger_infos["index_T2_crossing"]) + dict_trigger_infos["index_T1_crossing"]
        dict_trigger_infos["NC"] = n_T2_crossing

        # Calulate the peak value
        dict_trigger_infos["Q"] = (np.max(np.abs(period_after_T1_crossing[:j])) - baseline) / dict_trigger_infos["NC"]

        return dict_trigger_infos
    
    
    def trigger(self,
                trace,
                channels=[0,1]):
        '''
        The "master trigger function" of the FLT0 class.
        Performs the offline FLT-0 trigger for a given trace.

        Arguments
        ---------
        `trace`
        type        : np.ndarray[float]
        units       : ADC counts
        description : ADC trace, with shape (3,N_samples).

        `channels`
        type        : list[int] or np.ndarray[int]
        description : Channels on which to apply the trigger algorithm.
        
                                    
        Returns
        -------
        `flt0_results`
        type        : dict
        description : Results of the FLT0 trigger:
                      - `trigger_flags_ch`, np.ndarray[bool]: Flags of the trigger per channel, with shape (3,). Returns False by default.
                      - `trigger_time`, int: Time of the trigger [in SAMPLES!]. Defined as `t_period` + `t_overlap` for the first channel that was triggered.
        '''

        trigger_flags_ch = np.zeros(3,dtype=bool)
        trigger_times_ch = -1*np.ones(3,dtype=int)

        for channel in channels:
            logger.debug(f'channel {channel}')
            # Compute the relevant parameters for the trigger
            try:
                dict_trigger_info = self._extract_trigger_parameters(trace[channel],baseline=0)
            except ValueError:
                continue # if there are no parameters to extract, skip the trigger evaluation
            
            # Check if the trigger conditions are satisfied regarding the number of T2 crossings
            if dict_trigger_info['NC'] < self.trigger_config['nc_min'] or dict_trigger_info['NC'] > self.trigger_config['nc_max']:
                logger.debug(f'NC = {dict_trigger_info["NC"]} is too large/small')
                continue

            # Check if the trigger conditions are satisfied regarding the charge (typically not used)
            if dict_trigger_info['Q'] < self.trigger_config['q_min'] or dict_trigger_info['Q'] > self.trigger_config['q_max']:
                logger.debug(f'Q = {dict_trigger_info["Q"]} is too large/small')
                continue

            # Trigger time in a channel is defined as the moment when all trigger conditions are satisfied
            trigger_times_ch[channel]  = dict_trigger_info['index_T1_crossing'] + self.trigger_config['t_period']//2
            
            logger.debug(f'Found a trigger!')

        # Determine the trigger time of the FLT-0
        # Defined as the moment when all trigger conditions are satisfied + coincidence time window
        if np.any(trigger_times_ch > -1):
            first_ch_trigger_time = np.min(trigger_times_ch[trigger_times_ch>=0])
            trigger_time          = first_ch_trigger_time + self.trigger_config['t_overlap']//2

            # Check if we get coincident triggers in the different channels in the coincidence time window
            for i, trigger_time_ch in enumerate(trigger_times_ch):
                if trigger_time_ch >= first_ch_trigger_time and trigger_time_ch <= trigger_time:
                    trigger_flags_ch[i] = True
        else:
            trigger_time = -1

        # Save the flt0 results in a dictionary
        flt0_results = {'trigger_flags_ch':trigger_flags_ch,
                        'trigger_time':trigger_time}
        
        return flt0_results