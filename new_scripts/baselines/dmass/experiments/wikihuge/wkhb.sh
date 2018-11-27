#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:4
#SBATCH --partition=titanx
#SBATCH --job-name=wkhb
#SBATCH --output=wkhb.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g

# Load modules
module restore

# Run the job
srun python ../../model/train.py -ngpus 4 -bsize 50 -fw transformer -out wkhb -layer_drop 0.2 -op adagrad -lr 0.1 --mode wikihuge -nhl 4 -nel 4 -ndl 4 -lc True --min_count 4 -eval_freq 0 --it_train True -warm /zfs1/hdaqing/saz31/text_simplification_0330/wikihuge_bn/model/model.ckpt-1258101

