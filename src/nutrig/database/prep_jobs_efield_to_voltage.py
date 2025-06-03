#! /usr/bin/env python3

import os
import glob
import numpy as np

'''
© Pablo Correa, 13 December 2023

Description:
    Script to make .sh files for jobs to be submitted at CC-IN2P3,
    specfically to use `convert_efield_to_voltage.py`.
'''

# Directory for job executables, output, and error files
job_dir = '/pbs/home/p/pcorrea/jobs/'

# Software used for electric-field simulations
sim = 'zhaires'

# Template for job-related filenames
job_name_template = 'job_convert_efield_to_voltage_sim_{}_start_{}_end_{}'

# Input directory for `convert_efield_to_voltage.py`
# Contains electric-field simulations in GRANDroot format
input_dir = '/sps/grand/pcorrea/nutrig/sim/{}/efield/'.format(sim)

# Output directory for `convert_efield_to_voltage.py`
# Will contain ADC voltage simulations in GRANDroot format
output_dir = input_dir.replace('efield','voltage_RFv2_nut')


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
    

    # Get the number of input files to be processed
    input_files = sorted( glob.glob(input_dir+'*.root') )
    n_files     = len(input_files)

    # Set the number of files to be processed per job
    n_files_per_job = 500

    # Compute the number of jobs
    n_jobs = int( np.ceil( n_files/n_files_per_job ) )
    
    # Create the job submission files
    for i in range(n_jobs)[:]:
        start = i*n_files_per_job
        end   = (i+1)*n_files_per_job

        job_name = job_name_template.format(sim,start,end)
        print(start,end,job_name)

        # SLURM options
        slurm_text  = '\n\n#SBATCH --job-name={}'.format(job_name)
        slurm_text += '\n#SBATCH --output={}'.format(job_dir+'out/'+job_name+'.out')
        slurm_text += '\n#SBATCH --error={}'.format(job_dir+'err/'+job_name+'.err')
        slurm_text += '\n#SBATCH --ntasks=1'
        slurm_text += '\n#SBATCH --mem=2000'
        slurm_text += '\n#SBATCH --time=0-12:00:00'

        # convert_efield_to_python options
        script_text = '\n\npython3 /pbs/home/p/pcorrea/grand/nutrig/database/v1/convert_efield_to_voltage.py {} -o {} -s {} -e {}'
        script_text = script_text.format(input_dir,output_dir,start,end)

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