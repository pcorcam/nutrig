#!/bin/sh

#SBATCH --job-name=job_make_sig_database
#SBATCH --output=/pbs/home/p/pcorrea/jobs/out/job_make_sig_database_%a.out 
#SBATCH --error=/pbs/home/p/pcorrea/jobs/err/job_make_sig_database_%a.err
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --time=0-02:00:00
#SBATCH --array=0-10

conda activate /pbs/home/p/pcorrea/.conda/envs/grandlib_2409/
source /pbs/home/p/pcorrea/grand/grandlib/env/setup.sh
source /pbs/home/p/pcorrea/grand/nutrig/env/setup.sh

python3 /pbs/home/p/pcorrea/grand/nutrig/src/nutrig/database/make_sig_database.py -df $SLURM_ARRAY_TASK_ID
