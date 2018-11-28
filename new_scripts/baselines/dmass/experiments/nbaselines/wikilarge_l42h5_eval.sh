#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=l42h5eval
#SBATCH --output=l42h5eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/eval.py -ngpus 1 -bsize 100 -fw transformer -out l42h5 -layer_drop 0.0 -op adagrad -lr 0.1 --mode dressnew -nhl 0 -nel 4 -ndl 1 -nh 5 -lc True --min_count 4 -eval_freq 0
