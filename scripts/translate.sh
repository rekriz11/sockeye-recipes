#!/bin/bash
#
# Translate an input file with a Sockeye NMT model

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: translate.sh -p hyperparams.txt -i input -o output -e ENV_NAME [-d DEVICE] [-s]"
  errcho "Input is a source text file to be translated"
  errcho "Output is filename for target translations"
  errcho "ENV_NAME is the sockeye conda environment name"
  errcho "Device is optional and inferred from ENV"
  errcho "-s is optional and skips BPE processing on input source"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}

BEAM_SIZE=1

while getopts ":h?p:e:i:o:d:b:t:n:m:y:s" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    b) BEAM_SIZE=$OPTARG
      ;;
    t) OUTPUT_TYPE=$OPTARG
      ;;
    p) HYP_FILE=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
    i) INPUT_FILE=$OPTARG
      ;;
    o) OUTPUT_FILE=$OPTARG
      ;;
    d) DEVICE=$OPTARG
      ;;
    s) SKIP_SRC_BPE=1
      ;;
    n) NGRAM_BLOCK=$OPTARG
      ;;
    m) SINGLE_HYP_MAX=$OPTARG
      ;;
    y) BEAM_SIBLING_PENALTY=$OPTARG
      ;;
  esac
done

if [[ -z $HYP_FILE || -z $ENV_NAME || -z $INPUT_FILE || -z $OUTPUT_FILE ]]; then
    errcho "Missing arguments"
    show_help
    exit 1
fi

###########################################
# (0) Setup
# source hyperparams.txt to get text files and all training hyperparameters
check_file_exists $HYP_FILE
check_file_exists $INPUT_FILE
source $HYP_FILE
source activate $ENV_NAME

# options for cpu vs gpu training (may need to modify for different grids)
source $rootdir/scripts/get-device.sh $DEVICE ""

###########################################
# (1) Book-keeping
LOG_FILE=${OUTPUT_FILE}.log
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "Start translating: $datenow on $(hostname)"
echo "$0 $@"
echo "$devicelog"

###########################################
# (2) Translate!
subword=$rootdir/tools/subword-nmt/
max_input_len=100

if [ "$SKIP_SRC_BPE" == 1 ]; then
    ### Run Sockeye.translate, then de-BPE:
    echo "Directly translating source input without applying BPE"
    cat $INPUT_FILE | \
	python -m sockeye.translate --models $modeldir $device \
	--disable-device-locking \
	--beam-size $BEAM_SIZE \
  --max-input-len $max_input_len \
  --output-type $OUTPUT_TYPE \
  --beam-block-ngram $NGRAM_BLOCK \
  --single-hyp-max $SINGLE_HYP_MAX \
  --beam-sibling-penalty $BEAM_SIBLING_PENALTY | \
	sed -r 's/@@( |$)//g' > $OUTPUT_FILE 
else
    ### Apply BPE to input, run Sockeye.translate, then de-BPE ###
    echo "Apply BPE to source input"
    python $subword/apply_bpe.py --input $INPUT_FILE --codes $bpe_vocab_src | \
	python -m sockeye.translate --models $modeldir $device \
	--disable-device-locking \
  --beam-size $BEAM_SIZE \
	--max-input-len $max_input_len \
  --output-type $OUTPUT_TYPE | \
	sed -r 's/@@( |$)//g' > $OUTPUT_FILE
fi

##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End translating: $datenow on $(hostname)"
echo "==========================================="
