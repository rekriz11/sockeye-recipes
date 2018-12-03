#!/bin/bash
#
# Preprocess train and validation data with BPE

## standard settings, need not modify
bpe_minfreq=5


# (1) paths to files
subword='~/sockeye-recipes/tools/subword-nmt'

dmass_base=~/sockeye-recipes/new_scripts/baselines/dmass/data/newsela/
bpe_vocab_src=${dmass_base}train/vocab.src.bpe
bpe_vocab_tgt=${dmass_base}train/vocab.tgt.bpe

train_src=${dmass_base}train/train.aner.src
train_tgt=${dmass_base}train/train.aner.tgt
train_all=${dmass_base}train/train.aner.all

train_bpe_src=${dmass_base}train/train.aner.src.bpe
train_bpe_tgt=${dmass_base}train/train.aner.tgt.bpe
train_bpe_all=${dmass_base}train/train.aner.all.bpe

valid_src=${dmass_base}valid/valid.aner.src
valid_tgt=${dmass_base}valid/valid.aner.tgt

valid_bpe_src=${dmass_base}valid/valid.aner.src.bpe
valid_bpe_tgt=${dmass_base}valid/valid.aner.tgt.bpe

bpe_symbols_src=50000
bpe_symbols_tgt=50000


###########################################
# (2) BPE on source side
echo `date '+%Y-%m-%d %H:%M:%S'` "- Learning BPE on source and creating vocabulary: $bpe_vocab_src"
python ~/sockeye-recipes/tools/subword-nmt/learn_bpe.py \
--input $train_src --output $bpe_vocab_src --symbols $bpe_symbols_src --min-frequency $bpe_minfreq 

echo `date '+%Y-%m-%d %H:%M:%S'` "- Applying BPE, creating: ${train_bpe_src}, ${valid_bpe_src}" 
python ~/sockeye-recipes/tools/subword-nmt/apply_bpe.py \
--input $train_src --codes $bpe_vocab_src --output $train_bpe_src

python ~/sockeye-recipes/tools/subword-nmt/apply_bpe.py \
--input $valid_src --codes $bpe_vocab_src --output $valid_bpe_src


###########################################
# (3) BPE on target side
echo `date '+%Y-%m-%d %H:%M:%S'` "- Learning BPE on target and creating vocabulary: $bpe_vocab_tgt"
python ~/sockeye-recipes/tools/subword-nmt/learn_bpe.py \
--input $train_tgt --output $bpe_vocab_tgt --symbols $bpe_symbols_tgt --min-frequency $bpe_minfreq 

echo `date '+%Y-%m-%d %H:%M:%S'` "- Applying BPE, creating: ${train_bpe_tgt}, ${valid_bpe_tgt}" 
python ~/sockeye-recipes/tools/subword-nmt/apply_bpe.py \
--input $train_tgt --codes $bpe_vocab_tgt --output $train_bpe_tgt
python ~/sockeye-recipes/tools/subword-nmt/apply_bpe.py \
--input $valid_tgt --codes $bpe_vocab_tgt --output $valid_bpe_tgt

echo `date '+%Y-%m-%d %H:%M:%S'` "- Done with preprocess-bpe.sh"

###########################################
# (3) BPE on all
echo `date '+%Y-%m-%d %H:%M:%S'` "- Learning BPE on target and creating vocabulary: $bpe_vocab_tgt"
python ~/sockeye-recipes/tools/subword-nmt/learn_bpe.py \
--input $train_all --output $bpe_vocab_all --symbols $bpe_symbols_all --min-frequency $bpe_minfreq 
