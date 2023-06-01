#!/bin/bash -l

#SBATCH --job-name=FeatureAblation		# Name for your job
#SBATCH --comment="Testing Job"		# Comment for your job

#SBATCH --account=admalware		# Project account to run your job under
#SBATCH --partition=tier3

#SBATCH --output=%x_%j.out		# Output file
#SBATCH --error=%x_%j.err		# Error file

#SBATCH --time=2-00:00:00		# Time limit
#SBATCH --nodes=1			# How many nodes to run on
#SBATCH --ntasks=1			# How many tasks per node
#SBATCH --cpus-per-task=1		# Number of CPUs per task
#SBATCH --mem-per-cpu=10g		# Memory per CPU
#SBATCH --gres=gpu:p4:1


/home/lk3591/anaconda3/envs/MalConv2/bin/python explain.py --run --analyze --resume --config_file=./config_files/explain/FeatureAblation.ini

