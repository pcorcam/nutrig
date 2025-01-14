#! /usr/bin/env python3

import os
import glob
import numpy as np

'''
© Pablo Correa, 13 December 2023

Description:
    Script to make .sh files for jobs to be submitted at CC-IN2P3,
    specfically to use `get_sim_traces_from_psa.py`.
'''

# Directory for job executables, output, and error files
job_dir = '/pbs/home/p/pcorrea/jobs/'

# Software used for air-shower simulations
sim = 'zhaires'

# Primary cosmic-ray type to analyze
primary = 'proton'

# RF chain version used for voltage simulations
rf_chain = 'rfv2'

# ADC threshold used to select pulses
thresh = 30

# Data directory where noise traces are stored
noise_dir = '/sps/grand/data/gp13/GrandRoot/2024/02/'
#months    = ['01','02','03']

# Input directory for `add_noise_to_signal.py`
# Contains ADC traces of air-shower simulations obtained with `get_sim_traces_from_psa.py` in npz format
input_dir = '/sps/grand/pcorrea/nutrig/datasets/sig/{}_thresh_{}/'.format(rf_chain,thresh)

# Output directory for `get_sim_traces_from_psa.py`
# Will contain ADC traces in npz format
output_dir = input_dir


if __name__ == '__main__':
    # Make the job-related directories
    if os.path.exists(job_dir):
        print('WARNING: job directories already exist in:',job_dir)
    else:
        print(job_dir)
        cmd = 'mkdir --verbose -p ' + job_dir + 'exe ' + job_dir + 'out ' + job_dir + 'err'
        os.system(cmd)

    # Make the output directory
    if os.path.exists(output_dir):
        print('WARNING: output directory already exists:',output_dir)
    else:
        cmd = 'mkdir --verbose ' + output_dir
        os.system(cmd)
    
    
    # One job per input file
    input_files = sorted( glob.glob(input_dir+'sig_traces_adc_*_{}_*.npz'.format(primary)) )
    n_jobs      = len(input_files)

    for i, input_file in enumerate(input_files):
        job_name = input_file.split('/')[-1].replace('sig_traces_adc','job_add_noise_to_sig_traces_adc')
        print(i,job_name)

        # # Choose GP13 noise data uniformly between January-March 2024
        # if i < n_jobs/3:
        #     j = 0
        # elif i < 2*n_jobs/3:
        #     j = 1
        # else:
        #     j = 2

        # noise_dir = noise_dir.format(months[j])

        # SLURM options
        slurm_text  = '\n\n#SBATCH --job-name={}'.format(job_name)
        slurm_text += '\n#SBATCH --output={}'.format(job_dir+'out/'+job_name+'.out')
        slurm_text += '\n#SBATCH --error={}'.format(job_dir+'err/'+job_name+'.err')
        slurm_text += '\n#SBATCH --ntasks=1'
        slurm_text += '\n#SBATCH --mem=8000'
        slurm_text += '\n#SBATCH --time=0-12:00:00'

        # add_noise_to_signal options
        script_text = '\n\npython3 /pbs/home/p/pcorrea/grand/nutrig/scripts/add_noise_to_signal.py {} -o {} -s {} -n {}'
        script_text = script_text.format(input_file,output_dir,i,noise_dir)

        # Text for the sh file
        job_text  = '#!/bin/sh'
        job_text += slurm_text
        job_text += '\n\nconda activate /sps/grand/software/conda/grandlib_2304/'
        job_text += '\nsource /pbs/home/p/pcorrea/grand/grandlib/env/setup.sh'
        job_text += script_text

        # Write the job submission file
        job_sh = open(job_dir+'exe/'+job_name+'.sh','w')
        job_sh.write(job_text)
        job_sh.close()

        # Make it an executable file
        cmd = 'chmod +x ' + job_dir + 'exe/' + job_name + '.sh'
        os.system(cmd)

    print('*** FINISHED ***')