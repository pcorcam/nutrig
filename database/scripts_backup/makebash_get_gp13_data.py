#! /usr/bin/env python3

import os
import glob
import numpy as np

'''
© Pablo Correa, 13 December 2023

Description:
    Script to make .sh files for jobs to be submitted at CC-IN2P3,
    specfically to use `get_gp13_data.py`.
'''

# Directory for job executables, output, and error files
job_dir = '/pbs/home/p/pcorrea/jobs/'

# Data directory where noise traces are stored
# This is the input directory for `get_gp13_data.py`
input_dir = '/sps/grand/data/gp13/GrandRoot/2024/'

# Select files from January 16, 2024 --- April 24, 2024
dates = np.arange(20240116,20240131+1)
dates = np.hstack( (dates, np.arange(20240201,20240229+1) ) )
dates = np.hstack( (dates, np.arange(20240301,20240331+1) ) )
dates = np.hstack( (dates, np.arange(20240401,20240424+1) ) )

# One job for every 1 days, make sure last day is included
dates_jobs = dates[::1]
if dates_jobs[-1] is not dates[-1]+1:
    dates_jobs = np.hstack( (dates_jobs,dates[-1]+1) )

# Output directory for `get_gp13_data.py`
# Will contain traces in npz format
output_dir = '/sps/grand/pcorrea/nutrig/database/bkg/gp13_raw'

job_name_template = 'job_get_gp13_data_from_{}_to_{}'


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

    for i in range( len(dates_jobs)-1 ):
        start_date = dates_jobs[i]
        end_date   = dates_jobs[i+1]-1

        job_name = job_name_template.format(start_date,end_date)
        print(i+1,job_name)

        # SLURM options
        slurm_text  = '\n\n#SBATCH --job-name={}'.format(job_name)
        slurm_text += '\n#SBATCH --output={}'.format(job_dir+'out/'+job_name+'.out')
        slurm_text += '\n#SBATCH --error={}'.format(job_dir+'err/'+job_name+'.err')
        slurm_text += '\n#SBATCH --ntasks=1'
        slurm_text += '\n#SBATCH --mem=24000'
        slurm_text += '\n#SBATCH --time=0-12:00:00'

        # add_noise_to_signal options
        script_text = '\n\npython3 /pbs/home/p/pcorrea/grand/nutrig/database/scripts/get_gp13_data.py {} -o {} -s {} -e {}'
        script_text = script_text.format(input_dir,output_dir,start_date,end_date)

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