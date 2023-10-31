#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=aligntransformer
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
# Specify the output file
#SBATCH --output=sys_output.out
#SBATCH --error=sys_error.out
python hook_mistral.py