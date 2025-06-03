#! /usr/bin/env python3
'''
Module for the NUTRIG first-level trigger based on template fitting.
'''

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import os
import logging
import sys

import numpy as np
from scipy.optimize import curve_fit


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


###-###-###-###-###-###-###- GENERAL FUNCTIONS -###-###-###-###-###-###-###

def rms(trace,
        **kwargs):
    '''
    Computes the root-mean-square (RMS) of a trace.

    Arguments
    ---------
    `trace`
    type        : np.ndarray
    description : Trace(s) for which to compute the RMS.

    `**kwargs`
    description : Keyword arguments passed to `np.mean()` to process multidimensional arrays of traces.

    Returns
    -------
    `rms`
    type        : np.ndarray
    description : RMS of the trace.
    '''

    rms = np.sqrt( np.mean( trace**2,**kwargs ) )

    return rms


def normalize(trace,
              **kwargs):
    '''
    Normalizes a trace between [-1,1].

    Arguments
    ---------
    `trace`
    type        : np.ndarray
    description : Trace(s) to normalize.

    `**kwargs`
    description : Keyword arguments passed to `np.max()` to process multidimensional arrays of traces.

    Returns
    -------
    `norm_trace`
    type        : np.ndarray
    description : Normalized trace.
    '''

    norm_trace = trace / np.max( np.abs(trace),**kwargs )

    return norm_trace


###-###-###-###-###-###-###- TEMPLATE FLT CLASS -###-###-###-###-###-###-###

class TemplateFLT:
    '''
    Class to create TemplateFLT objects.
    The template FLT

    `self.templates`
    type        : np.ndarray
    description : Template array, with shape (self.n_templates,n_samples_template). Typically n_samples_template = 400.

    `self.pol`
    type        : str
    description : Polarization corresponding to the templates. Can be 'XY' or 'Z'.

    `self.rf_chain`
    type        : str
    description : Version of the RF chain corresponding to the templates. Format: 'rfv1', 'rfv2',...

    `self.n_templates`
    type        : int
    description : The total number of templates in `self.templates`.

    
    `self._adc_sampling_rate`
    type        : float
    unit        : MHz
    description : Sampling rate of the ADC. Default is 500 MHz (time resolution of 2 ns).

    `self._sim_sampling_rate`
    type        : float
    unit        : MHz
    description : Sampling rate of the simulation used to generate the templates. Default is 2 GHz (time resolution of 0.5 ns).

    `self._desampling_factor`
    type        : int
    description : Factor to desample templates from `self._sim_sampling_rate` to `self.adc_sampling_rate`. Default is 4.

    `self.templates_desampled`
    type        : np.ndarray
    description : Desampled-template array, with shape (self.n_templates,self._desampling_factor,n_sim_samples/self._desampling_factor).
                  Typically n_samples_template = 400 in units of 0.5 ns, so n_adc_s



    `self._template_peak_sample`
    type        : 
    unit        : ADC samples
    description : 

    '''
    def __init__(self):
        '''
        Initialiser of the TemplateFLT class.
        '''

        self.templates   = np.array([])
        self.pol         = ''
        self.rf_chain    = ''
        self.n_templates = len(self.templates)
        
        self._adc_sampling_rate   = 500  # [MHz]
        self._sim_sampling_rate   = 2000 # [MHz]
        self._desampling_factor   = int( self._sim_sampling_rate/self._adc_sampling_rate )
        self.templates_desampled  = np.array([])

        self._template_peak_sample = 30

        self._corr_window = np.array([-10,10],dtype=int) # ADC samples
        self._fit_window  = np.array([-10,30],dtype=int) # ADC samples
        self._ts_thresh   = 1

        self.corr_best                    = np.array([])
        self.time_best                    = np.array([])
        self.ampl_best                    = np.array([])
        self.chi2                         = np.array([]) # will be reduced chi2
        self.rss_post_peak                = np.array([]) # RSS = residual sum of squares (basically chi2)
        self.idx_templates_desampled_best = np.array([])

        self.idx_template_best_fit = 0
        self.ts                    = 0

        return


    def load_templates(self,
                       path_template_dir,
                       n_templates,
                       pol,
                       rf_chain,
                       random_temp=None):
        '''
        Loads in templates saved in a file of the following format:
        
        templates_pol_rf_chain.npz 
        
        See `TEMPLATE BLA IPYNB` for more details.

        Arguments
        ---------
        `path_template_dir`
        type        : str
        description : Path to the directory where the file containing the templates is stored.

        `pol`
        type        : str
        description : Specifies the polarization of the templates.

        `rf_chain`
        type        : str
        description : Specifies the version of the RF chain used to generate the templates.
        '''

        assert pol in ['XY','Z']
        assert rf_chain in ['rfv1','rfv2']

        if random_temp is not None:
            path_template_file  = os.path.join(path_template_dir,f'templates_random_{random_temp}_{n_templates}_{pol}_{rf_chain}.npz')
        else:
            path_template_file  = os.path.join(path_template_dir,f'templates_{n_templates}_{pol}_{rf_chain}.npz')

        template_file       = np.load(path_template_file)

        self.templates = template_file['templates']

        assert n_templates == len(self.templates)

        self.pol         = pol
        self.rf_chain    = rf_chain
        self.n_templates = n_templates

        self._desample_templates()

        logger.info(f'Loaded {n_templates} templates from {path_template_file}')

        return
    

    def _desample_templates(self):
        '''
        Creates a set of desampled templates for each of the original templates.

        We will generally use templates of 400 samples simulated with a time resolution of 0.5 ns (2 GHz).
        Data recorded by the ADC has a time resolution of 2 ns (500 MHz). In this case, self._desampling_factor = 4.
        Therefore, for each template, we can create 4 sub-templates of 100 samples, each with a time resolution of 0.5 ns.
        '''

        assert self._desampling_factor >= 1, 'Desampling factor has to be >= 1! Simulation must have a sampling rate >= ADC sampling rate!'

        len_templates_desampled = int( self.templates.shape[-1] / self._desampling_factor )

        templates_desampled = np.zeros( (self.n_templates,self._desampling_factor,len_templates_desampled) )

        for i in range(self._desampling_factor):
            templates_desampled[:,i,:] = self.templates[:,i::4]

        self.templates_desampled = templates_desampled

        return


    def set_sampling_rates(self,
                           adc_sampling_rate,
                           sim_sampling_rate):
        '''
        Setter function for self._sim_sampling_rate and self._adc_sampling_rate.

        Arguments:
        ----------
        `adc_sampling_rate`
        type        : float
        unit        : MHz
        description : Sampling rate of the ADC.

        `sim_sampling_rate`
        type        : float
        unit        : MHz
        description : Sampling rate of the template simulation. Must be a multiple >= `adc_sampling_rate`.
        '''
        
        assert sim_sampling_rate >= adc_sampling_rate, 'Simulation must have a sampling rate >= ADC sampling rate!'

        if sim_sampling_rate % adc_sampling_rate != 0:
            logger.warning(f'Template sampling rate is not a multiple of ADC sampling rate')

        self.sim_sampling_rate = sim_sampling_rate
        self.adc_sampling_rate = adc_sampling_rate
        self.desampling_factor = int( self.sim_sampling_rate/self.adc_sampling_rate )

        logger.debug(f'Desampling factor = {self.desampling_factor}')

        self._desample_templates()

        return

    
    def set_corr_window(self,
                        corr_window):
        '''
        Setter function for self.corr_window.

        Arguments:
        ----------
        `corr_window`
        type        : list[int] or np.ndarray[int]
        unit        : ADC samples (2 ns)
        description : Scan window for the cross correlation, relative to the time of trace maximum (after FLT-0) at 0.
        '''

        assert len(corr_window) == 2, 'Window must be of the format [window_start,window_end]!'
        assert corr_window[0] < 0 and corr_window[1] > 0, 'Window must be relative to trace maximum time at 0!'

        self._corr_window = np.array(corr_window,dtype=int)

        return

    
    def set_fit_window(self,
                       fit_window):
        '''
        Setter function for self.fit_window.

        Arguments:
        ----------
        `fit_window`
        type        : list[int] or np.ndarray[int]
        unit        : ADC samples (2 ns)
        description : Window for the template fit, relative to a best-correlation time at 0.
        '''

        assert len(fit_window) == 2, 'Window must be of the format [window_start,window_end]!'
        assert fit_window[0] < 0 and fit_window[1] > 0, 'Window must be relative to best-trigger time at 0!'

        self._fit_window = np.array(fit_window,dtype=int)

        return

    
    def set_ts_thresh(self,
                      ts_thresh):
        '''
        Setter function for self.ts_thresh.

        Arguments:
        ----------
        `ts_thresh`
        type        : float
        description : TS threshold to be used for the trigger decision.
        '''

        self._ts_thresh = ts_thresh

        return
    
    
    def _cross_corr_1_template(self,
                               trace,
                               template):
        '''
        Computes the cross correlation of a trace and a single template.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_trace_samples).

        `template`
        type        : np.ndarray[float]
        unit        : ADC counts
        description : Template (desampled) with a resolution of self._adc_sampling_rate, with shape (n_template_samples).

        Returns
        -------
        `corr`      : np.ndarray
        description : Cross correlation of the trace and template, with shape (n_trace_samples+n_template_samples+1).
        '''

        corr = np.correlate(trace,
                            template,
                            mode='valid')
        
        corr /= len(template) # normalizes between [-1,1]

        return corr
    

    def _cross_corr_desampled(self,
                              trace,
                              pre_trigger_time):
        '''
        Computes the cross correlation of a trace with all desampled templates.
        The cross correlation is computed in a window `self._corr_window` around `pre_trigger_time`.

        NOTE: Below, `corr_window_start` is the first sample for which the correlation is computed.
              This requires `len_templates_desampled` samples of the trace if we use np.correlate in 'valid' mode,
              as done in `self._cross_corr_1_template`.
              For sample i = 0,...,N, in the window, the computation of the correlation requires the samples
              corr_window_start + i, corr_window_start + i + 1, ..., corr_window_start + i + len_templates_desampled.
              Thus, below, `corr_window_end` corresponds to `corr_window_start + N + len_templates_desampled`.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_trace_samples).

        `pre_trigger_time`
        type        : int
        unit        : ADC samples
        description : Sample corresponding to a pre-trigger time.

        Returns:
        --------
        `corr_window_start`
        type        : int
        unit        : ADC samples
        description : Starting sample of the trace for the correlation window.

        `corr_desampled`
        type        : np.ndarray[float]
        description : Cross correlation of a trace and all sets of desampled templates,
                      with shape (self.n_templates,self.desampling_factor,corr_window_size+1).
                      See below for the exact defintion of `corr_window_size`.
        '''
        # fig bug of int
        pre_trigger_time = int(pre_trigger_time)
         
        len_templates_desampled = self.templates_desampled.shape[-1]
        #corr_window_size        = self._corr_window[1] - self._corr_window[0]
        t     = np.arange( trace.shape[-1] )[pre_trigger_time-5:-100]
        idx_0 = np.argmax( np.abs( trace[pre_trigger_time-5:-100] ) )
        t_0   = t[idx_0]

        # Make sure the window for the correlation scan is not outside the trace bounds
        # corr_window_start = int( np.max( [pre_trigger_time+self._corr_window[0],0] ) )
        # corr_window_end   = int( np.min( [pre_trigger_time+self._corr_window[1]+len_templates_desampled,len(trace)] ) )
        corr_window_start = int( np.max( [t_0+self._corr_window[0],0] ) )
        corr_window_end   = int( np.min( [t_0+self._corr_window[1]+len_templates_desampled,len(trace)] ) )

        # Only select the relevant samples of the trace in the correlation window
        t_window     = np.arange(corr_window_start,corr_window_end) - self._template_peak_sample
        trace_window = trace[t_window]

        # Calculate the sliding RMS of the trace to normalize the cross correlation
        sliding_trace_window = np.lib.stride_tricks.sliding_window_view(trace_window,len_templates_desampled)
        sliding_rms          = rms(sliding_trace_window,axis=1)
        print(sliding_rms)
        # Compute the cross correlation for all desampled templates
        corr_desampled = np.zeros( (self.n_templates,self._desampling_factor,sliding_rms.shape[0]) )

        for i, template_desampled_set in enumerate(self.templates_desampled):
            for j, template_desampled in enumerate(template_desampled_set):
                corr_desampled[i,j] = self._cross_corr_1_template(trace_window,template_desampled) / sliding_rms
        
        return corr_window_start, corr_desampled


    def _fit_time(self,
                  trace,
                  pre_trigger_time):
        '''
        Finds the best-fit time, in ADC samples, of a trace with all templates.
        For each template, the best-fit time is defined as the sample where the absolute value of the 
        cross correlation of the trace and the template is maximal.
        NOTE: For each template, we compute the cross correlation of its desampled templates with the trace.
              The desampled template that maximizes the abs(cross correlation) yields the best-fit time for that template.
              The index of the best desampled template is stored in `self.idx_templates_desampled_best`.
              This allows to fit the time of a pulse to the time resolution of `self._sim_sampling_rate`.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_trace_samples).

        `pre_trigger_time`
        type        : int
        unit        : ADC samples
        description : Sample corresponding to a pre-trigger time.
        '''
        
        corr_window_start, corr_desampled = self._cross_corr_desampled(trace,pre_trigger_time)

        corr_desampled_abs  = np.abs(corr_desampled)
        corr_desampled_sign = np.sign(corr_desampled)
        
        corr_best_desampled      = np.max(corr_desampled_abs,axis=2)# * corr_desampled_sign
        time_corr_best_desampled = np.argmax(corr_desampled_abs,axis=2)

        self.idx_templates_desampled_best = np.argmax(corr_best_desampled,axis=1)

        corr_best = np.zeros(self.n_templates)
        time_best = corr_window_start*np.ones(self.n_templates)

        for k, i in enumerate(self.idx_templates_desampled_best):
            corr_best[k]  = corr_best_desampled[k,i] * corr_desampled_sign[k,i,time_corr_best_desampled[k,i]]
            time_best[k] += time_corr_best_desampled[k,i]

        self.corr_best = corr_best
        self.time_best = time_best

        return


    def _fit_amplitude_1_template(self,
                                  trace, 
                                  template,
                                  **kwargs):
        '''
        Fits the amplitude of a single template (desampled) to a given trace.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_fit_samples).

        `template`
        type        : np.ndarray[float]
        unit        : ADC counts
        description : Template (desampled) with a resolution of self._adc_sampling_rate, with shape (n_fit_samples).

        `**kwargs`
        description : Keyword arguments passed to `scipy.optimize.curve_fit()`.

        Returns
        -------
        `ampl_best` : float
        description : Best-fit amplitude of a template to the given trace.

        `chi2`      : float
        description : Reduced chi2 computed for the best-fit solution.
        '''
        
        assert len(trace) == len(template), 'Trace and template need to be of same length in fitting window!'

        samples = np.arange(len(trace))

        # Define the fit function
        # Need to redefine `fit_samples` here so that `curve_fit` does not mess up with integer samples
        def _fit_func(samples,amplitude):
            fit_samples = np.array(samples,dtype=int)
            return amplitude*template[fit_samples]

        pbest, _, info, _, _ = curve_fit(_fit_func,
                                         samples,
                                         trace,
                                         full_output=True,
                                         **kwargs)

        ampl_best = pbest[0]

        res  = info['fvec']
        ndof = len(samples) - len(pbest)
        chi2 = np.sum( res**2 ) / ndof 

        rss_post_peak = np.sum( res[-self._fit_window[0]+15:]**2 )

        return ampl_best, chi2, rss_post_peak


    def _fit_amplitude(self,
                       trace):
        '''
        Finds the best-fit amplitude of all templates with respect to a trace.
        The fit is performed in a window `self._fit_window` around `self.time_best`.
        The points are weighted by `np.std(trace)**2 / trace` to give larger weights to samples with higher ADC counts.
        NOTE: the fit uses the desampled template that yielded the best time fit using `self._fit_time()`.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_trace_samples).
        '''
        
        # Only fit a part of the template (around the peak)
        samples_fit_window          = np.arange(self._fit_window[0],self._fit_window[1],1,dtype=int)
        samples_template_fit_window = samples_fit_window + self._template_peak_sample

        # Give larger weights to points with larger values
        # Patch to not devide by zero
        sigma_fit = np.abs( rms(trace)**2 / np.where(trace==0,1,trace) )

        ampl_best     = np.zeros(self.n_templates)
        chi2          = np.zeros(self.n_templates)
        rss_post_peak = np.zeros(self.n_templates)

        for k, i in enumerate( self.idx_templates_desampled_best ):
            samples_trace_fit_window = samples_fit_window + int(self.time_best[k])
            template_fit             = normalize(self.templates_desampled[k,i])
            
            ampl_best[k], chi2[k], rss_post_peak[k] = self._fit_amplitude_1_template(trace[samples_trace_fit_window],
                                                                                     template_fit[samples_template_fit_window],
                                                                                     #sigma=sigma_fit[samples_trace_fit_window],
                                                                                     )

        self.ampl_best     = ampl_best
        self.chi2          = chi2
        self.rss_post_peak = rss_post_peak

        return


    def _compute_ts(self):
        '''
        Computes the test statistic of the FLT.
        '''

        ts = np.abs( self.corr_best[self.idx_template_best_fit] )

        self.ts = ts

        return


    def template_fit(self,
                     trace,
                     pre_trigger_time):
        '''
        Performs the template fit given a pre-trigger time. 
        The template fit is performed in the following sequence:
        1. Fit the best pulse time for each template;
        2. Fit the best amplitude for each template;
        3. Determine which template yields the smallest reduced chi2. This is the best-fit template.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_trace_samples).

        `pre_trigger_time`
        type        : int
        unit        : ADC samples
        description : Sample corresponding to a pre-trigger time.
        '''

        assert self.n_templates > 0, 'No templates loaded! See `self.load_templates()`'

        self.corr_best                    = np.zeros(self.n_templates)
        self.time_best                    = np.zeros(self.n_templates)
        self.ampl_best                    = np.zeros(self.n_templates)
        self.chi                          = np.zeros(self.n_templates)
        self.idx_templates_desampled_best = np.zeros(self.n_templates)

        norm_trace = normalize(trace) # normalize the trace here

        logger.info('fitting time')
        self._fit_time(norm_trace,pre_trigger_time)

        logger.info('fitting amplitude')
        self._fit_amplitude(norm_trace)

        logger.info('computing ts')
        self.idx_template_best_fit = np.argmin(self.chi2)
        #self.idx_template_best_fit = np.argmax( np.abs(self.ampl_best) )
        self._compute_ts()

        return

    
    def trigger(self,
                trace,
                pre_trigger_time):
        '''
        The 'master trigger function' of the TemplateFLT class.
        For a given pre_trigger_time, a template fit is performed of the candidate pulse in a given trace.
        Subsequently, the test statistic is computed. If this test statistic is larger than
        `self._ts_thresh`, then the candidate pulse will be triggered.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_trace_samples).

        `pre_trigger_time`
        type        : int
        unit        : ADC samples
        description : Sample corresponding to a pre-trigger time.
        '''

        self.template_fit(trace,pre_trigger_time)

        if self.ts >= self._ts_thresh:
            decision = True
        else:
            decision = False
        
        return decision


class TemplateFLT3D:
    def __init__(self):
        '''
        Initialiser of the TemplateFLT3D class.
        Simply contains one TemplateFLT object for each polarization.
        '''

        self.FLT_X = TemplateFLT()
        self.FLT_Y = TemplateFLT()
        self.FLT_Z = TemplateFLT()

        self.ts = 0

        return
    
    def load_templates_X(self,
                         path_template_dir,
                         n_templates_X,
                         rf_chain,
                         **kwargs):
        '''
        Loads in templates saved in a file of the following format:
        
        templates_pol_rf_chain.npz 
        
        See `make_template_lib.ipynb` for more details.

        Arguments
        ---------
        `path_template_dir`
        type        : str
        description : Path to the directory where the files containing the templates are stored.

        `n_templates_X`
        type        : int
        description : Specifies the number of templates to load for the X polarization. `None` loads no templates in X.

        `rf_chain`
        type        : str
        description : Specifies the version of the RF chain used to generate the templates.
        '''
        
        self.FLT_X.load_templates(path_template_dir,n_templates_X,'XY',rf_chain,**kwargs)
        
        return
    
    
    def load_templates_Y(self,
                         path_template_dir,
                         n_templates_Y,
                         rf_chain,
                         **kwargs):
        '''
        Loads in templates saved in a file of the following format:
        
        templates_pol_rf_chain.npz 
        
        See `TEMPLATE BLA IPYNB` for more details.

        Arguments
        ---------
        `path_template_dir`
        type        : str
        description : Path to the directory where the files containing the templates are stored.

        `n_templates_Y`
        type        : int
        description : Specifies the number of templates to load for the Y polarization. `None` loads no templates in Y.

        `rf_chain`
        type        : str
        description : Specifies the version of the RF chain used to generate the templates.
        '''
        
        self.FLT_Y.load_templates(path_template_dir,n_templates_Y,'XY',rf_chain,**kwargs)
        
        return
    

    def load_templates_Z(self,
                         path_template_dir,
                         n_templates_Z,
                         rf_chain,
                         **kwargs):
        '''
        Loads in templates saved in a file of the following format:
        
        templates_pol_rf_chain.npz 
        
        See `TEMPLATE BLA IPYNB` for more details.

        Arguments
        ---------
        `path_template_dir`
        type        : str
        description : Path to the directory where the files containing the templates are stored.
        
        `n_templates_Z`
        type        : int
        description : Specifies the number of templates to load for the Z polarization. `None` loads no templates in Z.

        `rf_chain`
        type        : str
        description : Specifies the version of the RF chain used to generate the templates.
        '''
        
        self.FLT_Z.load_templates(path_template_dir,n_templates_Z,'Z',rf_chain,**kwargs)
        
        return
    
    
    def set_corr_window(self,
                        corr_window):
        '''
        Sets the correlation window for all 3 three channels.

        Arguments:
        ----------
        `corr_window`
        type        : list[int] or np.ndarray[int]
        unit        : ADC samples (2 ns)
        description : Scan window for the cross correlation, relative to a pre-trigger time at 0.
        '''

        self.FLT_X.set_corr_window(corr_window)
        self.FLT_Y.set_corr_window(corr_window)
        self.FLT_Z.set_corr_window(corr_window)

        return

    
    def set_fit_window(self,
                       fit_window):
        '''
        Sets the fit window for all 3 three channels.

        Arguments:
        ----------
        `fit_window`
        type        : list[int] or np.ndarray[int]
        unit        : ADC samples (2 ns)
        description : Window for the template fit, relative to a best-correlation time at 0.
        '''

        self.FLT_X.set_fit_window(fit_window)
        self.FLT_Y.set_fit_window(fit_window)
        self.FLT_Z.set_fit_window(fit_window)

        return
    

    def template_fit(self,
                     trace,
                     pre_trigger_time,
                     pre_trigger_flag):
        '''
        Performs the template fit for all specified polarizations.
        
        The template fit is performed in the following sequence:
        1. Fit the best pulse time for each template;
        2. Fit the best amplitude for each template;
        3. Determine which template yields the smallest reduced chi2. This is the best-fit template.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_pol,n_trace_samples).

        `pre_trigger_time`
        type        : int
        unit        : ADC samples
        description : Sample corresponding to a pre-trigger time.

        `pre_trigger_flag`
        type        : int
        description : Flag corresponding to the channels were there was a successful pretrigger.
                      X = 1, Y = 2, Z = 3, XY = 12, XZ = 13, YZ = 23, XYZ = 123
        '''

        ts = 0

        # X = 1, Y = 2, Z = 3, XY = 12, XZ = 13, YZ = 23, XYZ = 123
        assert pre_trigger_flag in [1,2,3,12,13,23,123], 'Pre-trigger flag does not correspond to a correct polarization!'

        if pre_trigger_flag in [1,12,13,123]:
            logger.info('Fitting X...')
            self.FLT_X.template_fit(trace[0],pre_trigger_time)

        if pre_trigger_flag in [2,12,23,123]:
            logger.info('Fitting Y...')
            self.FLT_Y.template_fit(trace[1],pre_trigger_time)
            
        if pre_trigger_flag in [3,13,23,123]:
            logger.info('Fitting Z...')
            self.FLT_Z.template_fit(trace[2],pre_trigger_time)

        ts      = np.max( [self.FLT_X.ts,self.FLT_Y.ts,self.FLT_Z.ts] )
        self.ts = ts
        
        return
    

    def trigger(self,
                trace,
                pre_trigger_time,
                pre_trigger_flag):
        '''
        The 'master trigger function' of the TemplateFLT class.
        For a given pre_trigger_time, a template fit is performed of the candidate pulse in a given trace.
        Subsequently, the test statistic is computed. If this test statistic is larger than
        `self._ts_thresh`, then the candidate pulse will be triggered.

        Arguments:
        ----------
        `trace`
        type        : np.ndarray[int]
        unit        : ADC counts
        description : Data trace with a resolution of self._adc_sampling_rate, with shape (n_trace_samples).

        `pre_trigger_time`
        type        : int
        unit        : ADC samples
        description : Sample corresponding to a pre-trigger time.

        `pol`
        type        : tuple[str]
        description : Polarizations for which to do the template fit. `pol` can only contain 'X', 'Y', or 'Z'.
        '''

        self.template_fit(trace,pre_trigger_time,pre_trigger_flag)

        if self.ts >= self._ts_thresh:
            decision = True
        else:
            decision = False
        
        return decision