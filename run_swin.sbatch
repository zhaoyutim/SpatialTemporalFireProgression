#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  100GB
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 4

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/zhao2/mambaforge/etc/profile.d/conda.sh
conda activate torchenv
PYTHONUNBUFFERED=1; python3 main.py -m $1 -b $2 -r $3 -lr $4 -nh $5 -ed $6 -av $7
