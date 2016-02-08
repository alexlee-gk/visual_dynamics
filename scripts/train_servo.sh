#!/bin/bash

if [ "$#" -eq 0 ]; then
	X1_C_DIM=32
	SHARE=1
elif [ "$#" -eq 1 ]; then
	X1_C_DIM=$1
	SHARE=1
elif [ "$#" -eq 2 ]; then
	X1_C_DIM=$1
	SHARE=$2
else
    echo "Usage: ./train_servo.sh [X1_C_DIM [SHARE]]"
    exit
fi
echo "X1_C_DIM=${X1_C_DIM} SHARE=${SHARE}"

N=0
T=0
DATA=data/servo32_train_data_200k.h5
VAL_DATA=data/servo32_val_data_0.h5
MAX_ITER=10000
ARGS="${DATA} ${VAL_DATA} --share ${SHARE} --x1_c_dim ${X1_C_DIM} --lr 0.001 -p fcn_action_cond_encoder_net -t $T -n $N -v0 --max_iter ${MAX_ITER} --concat 1"
SIM_ARGS="servo"

# ablated, one bilinear connection
time python servoing_controller.py ${ARGS} --levels 3 --ladder 0 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 2 --ladder 0 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 1 --ladder 0 ${SIM_ARGS}

# same as above, but with ladder loss. initialize with the respective networks from above
time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 --pf "models/caffe/FcnActionCondEncoderNet_levels3_x1cdim${X1_C_DIM}_numds0_share${SHARE}_ladder0_bn0_concat1_servo32_lr0.001/snapshot/_iter_${MAX_ITER}.caffemodel" ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 2 --ladder 1 --pf "models/caffe/FcnActionCondEncoderNet_levels2_x1cdim${X1_C_DIM}_numds0_share${SHARE}_ladder0_bn0_concat1_servo32_lr0.001/snapshot/_iter_${MAX_ITER}.caffemodel" ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 1 --ladder 1 --pf "models/caffe/FcnActionCondEncoderNet_levels1_x1cdim${X1_C_DIM}_numds0_share${SHARE}_ladder0_bn0_concat1_servo32_lr0.001/snapshot/_iter_${MAX_ITER}.caffemodel" ${SIM_ARGS}

# pixel-level at various resolutions. ladder loss is not applicable here
time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 1 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 2 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 3 ${SIM_ARGS}

# curriculum learning for our network. initialize with the network from the level above
time python servoing_controller.py ${ARGS} --levels 2 3 --ladder 1 --pf levels3 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 1 2 3 --ladder 1 --pf levels23 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 0 1 2 3 --ladder 1 --pf levels123 ${SIM_ARGS}
