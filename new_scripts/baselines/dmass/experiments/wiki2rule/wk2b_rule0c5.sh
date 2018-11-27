#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=wk2b_rule0c5
#SBATCH --output=wk2b_rule0c5.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/train.py -ngpus 1 -bsize 50 -fw transformer -out wk2b_rule0c5 -layer_drop 0.2 -op adagrad -lr 0.1 --mode dressnew --dmode v2 -nhl 4 -nel 4 -ndl 4 -lc True --min_count 4 -eval_freq 0 --memory rule -memcfg mofinal_mincnt5 --memory_prepare_step 30000 -warm /zfs1/hdaqing/saz31/text_simplification_0424/wk2b/ckpt/model.ckpt-197607

