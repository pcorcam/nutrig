#! /usr/bin/env python3

import os
import glob
import numpy as np

'''
© Pablo Correa, 13 December 2023

Description:
    Script to make .sh files for jobs to be submitted at CC-IN2P3,
    specfically to use `make_sig_database.py`.
'''

# Directory for job executables, output, and error files
job_dir = '/pbs/home/p/pcorrea/jobs/'

# Some parameters of the simulations
sim_soft = 'zhaires'
rf_chain = 'rfv2'
primary  = 'Proton'
thresh1  = 35
thresh2  = 25

# Data directory where simulated traces are stored in GrandRoot format
# Contains the input files for `make_sig_database.py`
input_dir   = f'/sps/grand/pcorrea/nutrig/sim/{sim_soft}/voltage_{rf_chain}/'
input_files = sorted( glob.glob( os.path.join(input_dir,f'*_{primary}_*.root') ) )

# Output and noise directory for `make_sig_database.py`
# Will contain traces in npz format
output_dir = f'/sps/grand/pcorrea/nutrig/database/sig/{sim_soft}_{primary.lower()}_{rf_chain}_th1_{thresh1}_th2_{thresh2}'
noise_dir  = f'/sps/grand/pcorrea/nutrig/database/bkg/gp13_pretrigger_stationary_th1_{thresh1}_th2_{thresh2}'

job_name_template = 'job_make_sig_database_{}'
n_files_per_job   = 100
n_files           = len(input_files)
n_jobs            = int( n_files/n_files_per_job ) + 1

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

    file_idx = 0
    for i in range(n_jobs):
        job_name = job_name_template.format(i)
        print(job_name)

        # SLURM options
        slurm_text  = '\n\n#SBATCH --job-name={}'.format(job_name)
        slurm_text += '\n#SBATCH --output={}'.format(job_dir+'out/'+job_name+'.out')
        slurm_text += '\n#SBATCH --error={}'.format(job_dir+'err/'+job_name+'.err')
        slurm_text += '\n#SBATCH --ntasks=1'
        slurm_text += '\n#SBATCH --mem=8000'
        slurm_text += '\n#SBATCH --time=0-12:00:00'

        script_text = ''
        for j in range(n_files_per_job):
            try:
                script_text_new = '\n\npython3 /pbs/home/p/pcorrea/grand/nutrig/database/make_sig_database.py {} -o {} -n {} -s {}'
                script_text    += script_text_new.format(input_files[file_idx],output_dir,noise_dir,file_idx)
                file_idx       += 1
            except:
                continue

        # Text for the sh file
        job_text  = '#!/bin/sh'
        job_text += slurm_text
        job_text += '\n\nconda activate /sps/grand/software/conda/grandlib_2304/'
        job_text += '\nsource /pbs/home/p/pcorrea/grand/grandlib/env/setup.sh'
        job_text += '\nsource /pbs/home/p/pcorrea/grand/nutrig/env/setup.sh'
        job_text += script_text

        # Write the job submission file
        job_sh = open(job_dir+'exe/'+job_name+'.sh','w')
        job_sh.write(job_text)
        job_sh.close()

        # Make it an executable file
        cmd = 'chmod +x ' + job_dir + 'exe/' + job_name + '.sh'
        os.system(cmd)

    print('*** FINISHED ***')