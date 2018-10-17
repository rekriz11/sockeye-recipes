##############################
## Predicting word complexity
##############################

cd ~/sockeye-recipes/new_scripts/

## Trains linear regression
## NOTE: I generated these features on nlpgrid, and copied them over.

python3 train_regression.py \
/data2/text_simplification/other_data/word_complexity/all_features_we3_weight0.4_0.7.txt \
Linear \
~/sockeye-recipes/new_scripts/lin_reg_we_weight_0.4_0.7.pkl