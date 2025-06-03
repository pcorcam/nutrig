#! /usr/bin/env python3

import os
import numpy as np

'''
© Pablo Correa, 13 December 2023

Description:
    Script to make .sh files for jobs to be submitted at CC-IN2P3,
    specfically to use `do_template_FLT_3D.py`.
'''

# Directory for job executables, output, and error files
job_dir = '/pbs/home/p/pcorrea/jobs/'

# Data file inputs to `do_template_FLT_3D.py`
bkg_train_file = '/sps/grand/pcorrea/nutrig/database/bkg/bkg_dataset_nutrig_gp13_train_seed_300.npz'
sig_train_file = '/sps/grand/pcorrea/nutrig/database/sig/sig_dataset_nutrig_gp13_train_seed_300.npz'

# Parameter spaces to cover
corr_windows_0 = [-25,-20,-15,-10,-5]
corr_windows_1 = [25,20,15,10,5]

fit_windows_0, fit_windows_1 = np.meshgrid([-15,-10,-5],[25,20,15,10,5])

n_templates_vals = [1,2,8,17,37,96]

# Directory where FLT results will be stored in npz format
output_dir = '/sps/grand/pcorrea/nutrig/template/results/'

# Templates for job files
job_name_template    = 'job_do_template_FLT_3D_templates_{}_cw_{}_{}_fw_{}_{}'
script_text_template = '\n\npython3 /pbs/home/p/pcorrea/grand/nutrig/flt/do_template_FLT_3D.py {} -o {} -nx {} -ny {} -cw {} {} -fw {} {} -v {}'


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

    i = 1
    for n_templates in n_templates_vals:
        for cw0, cw1 in zip(corr_windows_0,corr_windows_1):
            for fw0, fw1 in zip(fit_windows_0.flatten(),fit_windows_1.flatten()):
                job_name = job_name_template.format(n_templates,cw0,cw1,fw0,fw1)
                print(i+1,job_name)
                i += 1

                # SLURM options
                slurm_text  = '\n\n#SBATCH --job-name={}'.format(job_name)
                slurm_text += '\n#SBATCH --output={}'.format(job_dir+'out/'+job_name+'.out')
                slurm_text += '\n#SBATCH --error={}'.format(job_dir+'err/'+job_name+'.err')
                slurm_text += '\n#SBATCH --ntasks=1'
                slurm_text += '\n#SBATCH --mem=4000'
                slurm_text += '\n#SBATCH --time=0-4:00:00'
                #slurm_text += '\n#SBATCH --partition=flash'

                # Fill in the parser arguments for the script
                script_text  = script_text_template.format(bkg_train_file,output_dir,n_templates,n_templates,cw0,cw1,fw0,fw1,'warning')
                script_text += script_text_template.format(sig_train_file,output_dir,n_templates,n_templates,cw0,cw1,fw0,fw1,'warning')

                # Text for the sh file
                job_text  = '#!/bin/sh'
                job_text += slurm_text
                job_text += '\n\nconda activate /sps/grand/software/conda/grandlib_2304/'
                #job_text += '\nsource /pbs/home/p/pcorrea/grand/grandlib/env/setup.sh'
                job_text += '\nsource /pbs/home/p/pcorrea/grand/nutrig/env/setup.sh'
                job_text += script_text

                # Write the job submission file
                job_sh = open(job_dir+'exe/'+job_name+'.sh','w')
                job_sh.write(job_text)
                job_sh.close()

                # Make it an executable file
                cmd = 'chmod +x ' + job_dir + 'exe/' + job_name + '.sh'
                os.system(cmd)