#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=wk2bbr
#SBATCH --output=wk2bbr.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/train.py -ngpus 1 -bsize 50 -fw transformer -out wk2bbr -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress --dmode v2 -nhl 4 -nel 4 -ndl 4 -lc True --min_count 5 -eval_freq 0

