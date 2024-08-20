#!/bin/bash
#SBATCH --time=0-01:00
#SBATCH --mem=10GB
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:4
#SBATCH -o output/jobid_%j.out
#SBATCH -e error/jobid_%j.err
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.13-fasrc01
source activate env
cd ..

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr-ib
export WANDB_RUN_NAME=$SLURM_JOB_ID

torchrun --standalone --nproc_per_node=4 gpt.py
