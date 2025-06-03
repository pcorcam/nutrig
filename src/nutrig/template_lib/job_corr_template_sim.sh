#!/bin/sh                                                                                                                                                                                                                                                                       

#SBATCH --job-name=job_corr_template_sim                                                                                                                                                                                                            
#SBATCH --output=/pbs/home/p/pcorrea/jobs/out/job_corr_template_sim.out                                                                                                                                                                                                              
#SBATCH --error=/pbs/home/p/pcorrea/jobs/err/job_corr_template_sim.err                                                                                                                                                                                                               
#SBATCH --ntasks=1                                                                                                                                                                                                                                                              
#SBATCH --mem=8000                                                                                                                                                                                                                                                              
#SBATCH --time=0-02:00:00                                                                                                                                                                                                                                                       
#SBATCH --array=0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000                                                                                                                                                                                                                                                            

conda activate /sps/grand/software/conda/grandlib_2304/
source /pbs/home/p/pcorrea/grand/grandlib/env/setup.sh
source /pbs/home/p/pcorrea/grand/nutrig/env/setup.sh

python3 /pbs/home/p/pcorrea/grand/nutrig/template_lib/v1/corr_templates_sim.py /sps/grand/pcorrea/nutrig/sim/v1/zhaires/ -s ${SLURM_ARRAY_TASK_ID}

