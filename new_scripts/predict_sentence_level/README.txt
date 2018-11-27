python3 run_new.py \
--mode train \
--model rand \
--datafile /data1/reno/newsela_splits/newsela_sents \
--save_model \
--epoch 20 \
--early_stopping \
--gpu 0

python3 run_new.py \
--mode train \
--model static \
--datafile /data1/reno/newsela_splits/newsela_sents \
--save_model \
--epoch 20 \
--early_stopping \
--gpu 1

python3 run_new.py \
--mode train \
--model non-static \
--datafile /data1/reno/newsela_splits/newsela_sents_old \
--save_model \
--epoch 20 \
--early_stopping \
--gpu 0

python3 run_new.py \
--mode train \
--model multichannel \
--datafile /data1/reno/newsela_splits/newsela_sents \
--save_model \
--epoch 20 \
--early_stopping \
--gpu 1

###################
## Testing Models
###################

python3 run_new.py \
--mode test \
--model rand \
--datafile /data1/reno/newsela_splits/newsela_sents_old \
--epoch 20 \
--gpu 0

python3 run_new.py \
--mode test \
--model static \
--datafile /data1/reno/newsela_splits/newsela_sents_lem \
--epoch 20 \
--gpu 1

python3 run_new.py \
--mode test \
--model non-static \
--datafile /data1/reno/newsela_splits/newsela_sents_old \
--epoch 20 \
--gpu 0 \
—-outfile /data1/reno/newsela_splits/newsela_sents_examples

python3 run_new.py \
--mode test \
--model multichannel \
--datafile /data1/reno/newsela_splits/newsela_sents_lem \
--epoch 20 \
--gpu 1