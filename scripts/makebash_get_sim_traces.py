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

# Trace length for the ADC traces
trace_length = 1024

# Input directory for `get_sim_traces_from_psa.py`
# Contains pulse-shape parameters obtained with `get_pulse_shape_params.py` in npz format
input_dir = '/sps/grand/pcorrea/nutrig/template/pulse_shape_analysis/{}_thresh_{}/'.format(rf_chain,thresh)

# Output directory for `get_sim_traces_from_psa.py`
# Will contain ADC traces in npz format
output_dir = '/sps/grand/pcorrea/nutrig/datasets/sig/{}_thresh_{}/'.format(rf_chain,thresh)


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
    input_files = sorted( glob.glob(input_dir+'*_{}_*.npz'.format(primary)) )

    for i, input_file in enumerate(input_files):
        job_name = input_file.split('/')[-1].replace('pulse_shape_params','job_get_sim_traces')
        print(i,job_name)

        # SLURM options
        slurm_text  = '\n\n#SBATCH --job-name={}'.format(job_name)
        slurm_text += '\n#SBATCH --output={}'.format(job_dir+'out/'+job_name+'.out')
        slurm_text += '\n#SBATCH --error={}'.format(job_dir+'err/'+job_name+'.err')
        slurm_text += '\n#SBATCH --ntasks=1'
        slurm_text += '\n#SBATCH --mem=8000'
        slurm_text += '\n#SBATCH --time=0-12:00:00'

        # get_sim_traces_from_psa options
        script_text = '\n\npython3 /pbs/home/p/pcorrea/grand/nutrig/scripts/get_sim_traces_from_psa.py {} -o {} -tl {}'
        script_text = script_text.format(input_file,output_dir,trace_length)

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