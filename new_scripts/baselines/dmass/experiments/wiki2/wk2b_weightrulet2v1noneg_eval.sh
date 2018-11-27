#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=wk2b_wrt2v1nn_eval
#SBATCH --output=wk2b_wrt2v1nn_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=long

# Load modules
module restore

# Run the job
srun python ../../model/eval.py -ngpus 1 -bsize 500 -fw transformer -out wk2b_wrt2v1nn -layer_drop 0.0 -op adagrad -lr 0.1 --mode dressnew --dmode v2 -nhl 4 -nel 4 -ndl 4 -lc True --min_count 4 -eval_freq 0
