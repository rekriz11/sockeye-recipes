#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=wiki_hugers
#SBATCH --output=wiki_hugers.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g

# Load modules
module restore

# Run the job
srun python ../../model/train.py -ngpus 1 -bsize 100 -fw transformer -out wiki_rs -layer_drop 0.0 -op adagrad -lr 0.1 --mode dressnew -nhl 4 -nel 4 -ndl 4 -lc True --min_count 4 -eval_freq 0 --memory rule -memcfg cffn_mofinal_mincnt5

