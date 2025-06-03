#!/bin/sh                                                                                                                                                                                                                                                                       

#SBATCH --job-name=job_corr_template_sim                                                                                                                                                                                                            
#SBATCH --output=/pbs/home/p/pcorrea/jobs/out/job_corr_template_sim.out                                                                                                                                                                                                              
#SBATCH --error=/pbs/home/p/pcorrea/jobs/err/job_corr_template_sim.err                                                                                                                                                                                                               
#SBATCH --ntasks=1                                                                                                                                                                                                                                                              
#SBATCH --mem=4000                                                                                                                                                                                                                                                              
#SBATCH --time=0-02:00:00                                                                                                                                                                                                                                                       
#SBATCH --array=0-95                                                                                                                                                                                                                                                        

conda activate /sps/grand/software/conda/grandlib_2409/
source /pbs/home/p/pcorrea/grand/grandlib/env/setup.sh
source /pbs/home/p/pcorrea/grand/nutrig/env/setup.sh

python3 /pbs/home/p/pcorrea/grand/nutrig/flt/do_template_FLT_3D.py /sps/grand/pcorrea/nutrig/database/sig/sig_dataset_bias_test_bins_10x10_zenith_30.6_87.35_omega_diff_0.0_2.0_seed_300.npz -o /sps/grand/pcorrea/nutrig/template/results_bias/ -t /sps/grand/pcorrea/nutrig/template/lib_random/ -nx 1 -ny 1 -r ${S\
LURM_ARRAY_TASK_ID} -cw -10 10 -fw -10 30 -v warning

