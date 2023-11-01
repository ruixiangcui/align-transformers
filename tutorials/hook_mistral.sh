#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=aligntransformer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=sys_output.out
#SBATCH --error=sys_error.out
python hook_mistral.py