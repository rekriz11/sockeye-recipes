#!/bin/bash

#echo "0. Copying data and embeddings ... Note that you need to request Newsela data"
#mkdir -p data/newsela_Zhang_Lapata_splits data/emb
#cp /data1/reno/newsela_Zhang_Lapata_splits/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.* data/newsela_Zhang_Lapata_splits
#cp /data2/text_simplification/dataset/test/* /data2/text_simplification/dataset/train/* /data2/text_simplification/dataset/valid/* data/newsela_Zhang_Lapata_splits
#cp /data1/embeddings/eng/glove.6B.300d.txt data/emb/glove.6B.300d.txt

hyperparam_file=anon_glove_loss_2.0.hpm
source $hyperparam_file
echo "1. Get rootdir from hyperparam file: $rootdir"

#echo "2. Running preprocess-BPE.sh"
#bash $rootdir/scripts/preprocess-bpe.sh $hyperparam_file

echo "3. Starting training with pre-trained embeddings"
bash $rootdir/scripts/train-embeddings_loss.sh -p $hyperparam_file -e sockeye_gpu_57
#qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=240:00:00,num_proc=2 -j y $rootdir/scripts/train.sh -p $hyperparam_file -e sockeye_gpu

# TBD: testset=newstest2014
echo "4. Decode is TBD"
# TBD:  bash $rootdir/scripts/translate.sh -p $hyperparam_file -i $testset.en -o $testset.de.1best -e sockeye_gpu
#qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=2:00:00 -j y $rootdir/scripts/translate.sh -p $hyperparam_file -i $testset.en -o $testset.de.1best -e sockeye_gpu

echo "5. Compute BLEU: TBD"
echo "$rootdir/tools/multi-bleu.perl $testset.de < $testset.de.1best 2> /dev/null"
