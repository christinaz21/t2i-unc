#!/bin/bash
#SBATCH --exclude=neu306
#SBATCH --job-name=images           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/n/fs/reas-llm-uq/t2i-unc/output/logs/test-%J.log
#SBATCH --error=/n/fs/reas-llm-uq/t2i-unc/output/logs/test-%J.log
# SBATCH --mail-type=all
# SBATCH --mail-user=christinaz@princeton.edu

module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"  # this is needed to load python packages correctly
conda activate oasis
module load cudatoolkit/12.4

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0
# export CUDA_LAUNCH_BLOCKING=1  # only for debugging slowdowns

CUDA_LAUNCH_BLOCKING=1 

python src/experiments/generate_all_images.py --model pixart --prompt_files "moderate.jsonl" "underspecified.jsonl"
