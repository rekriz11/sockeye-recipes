#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=l3h2
#SBATCH --output=l3h2.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=normal
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/train.py -ngpus 1 -bsize 100 -fw transformer -out l3h2 -layer_drop 0.2 -op adagrad -lr 0.1 --mode dressnew -nhl 3 -nel 3 -ndl 3 -nh 2 -lc True --min_count 4 -eval_freq 0

