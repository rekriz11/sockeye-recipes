#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:4
#SBATCH --partition=gtx1080
#SBATCH --job-name=wk2bc5
#SBATCH --output=wk2bc5.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/train.py -ngpus 4 -bsize 50 -fw transformer -out wk2b -layer_drop 0.2 -op adagrad -lr 0.1 --mode dressnew --dmode v2 -nhl 4 -nel 4 -ndl 4 -lc True --min_count 5 -eval_freq 0

