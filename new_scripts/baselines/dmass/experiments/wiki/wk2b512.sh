#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:4
#SBATCH --partition=gtx1080
#SBATCH --job-name=wk2bb512
#SBATCH --output=wk2bb512.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/train.py -ngpus 4 -bsize 16 -fw transformer -out wk2bb512 -dim 512 -layer_drop 0.2 -op adam -lr 0.001 --mode dress -nhl 4 -nel 4 -ndl 4 -lc True --min_count 5 -eval_freq 0 -nh 8 -warm /zfs1/hdaqing/saz31/text_simplification_0504/wk2bb512/log2/model.ckpt-36757

