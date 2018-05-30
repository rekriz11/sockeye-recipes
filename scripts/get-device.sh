#!/bin/bash

# Select device to use
if [ "$1" == "cpu" ]; then
    # Use CPU
    device="--use-cpu"
    log="get-device.sh: On $HOSTNAME Using CPU. `date`"
else
    # Use GPU
    module load cuda90/toolkit

    if [ $CUDA_VISIBLE_DEVICES ]; then
	# if CUDA_VISIBLE_DEVICES is already set, just return 0
	# TODO: support multiGPU
	device="--device-id 0"
	log="get-device.sh: On $HOSTNAME CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES found, so set device_id = 0. `date`"
    else
	# else, look for a free GPU
	freegpu=( $( nvidia-smi --format=noheader,csv --query-gpu=index,utilization.gpu | grep ', 0 %' | awk -F ',' '{print $1}' ) )

	numfree=${#freegpu[@]}
	
	if [ $numfree -gt 0 ]; then
	    # randomly pick one gpu id to return
	    pick=$(expr $RANDOM % $numfree )
	    device="--device-id ${freegpu[$pick]}"
	    echo "get-gpu.sh: Picking device_id = ${freegpu[$pick]} from free GPUs: ${freegpu[@]}. `date`"
	else
	    # No GPUs, default back to CPU
	    device="--use-cpu"
	    log="get-gpu.sh: On $HOSTNAME Using CPU because no free GPUs. `date`"
	fi
    fi
fi

echo $device
echo $log

