#!/bin/bash

if [ "$#" -eq 0 ]; then
	FACTOR=1
elif [ "$#" -eq 1 ]; then
	FACTOR=$1
else
    echo "Usage: ./run_experiments_servo.sh [FACTOR]"
    exit
fi
echo "FACTOR=${FACTOR}"

X1_C_DIM=32
SHARE=1
N=20
T=20
DATA=data/servo32_train_data_200k.h5
VAL_DATA=data/servo32_val_data_0.h5
MAX_ITER=10000
ARGS="${DATA} ${VAL_DATA} --share ${SHARE} --x1_c_dim ${X1_C_DIM} --lr 0.001 -p fcn_action_cond_encoder_net -t $T -n $N -v0 --max_iter ${MAX_ITER} --concat 1 --pf auto --no_train --experiment 0 --dof_limit_factor ${FACTOR} --output_results_dir results"
SIM_ARGS="servo"

# ablated, one bilinear connection (with ladder loss)
time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 2 --ladder 1 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 1 --ladder 1 ${SIM_ARGS}

# pixel-level at various resolutions. ladder loss is not applicable here
time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 1 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 2 ${SIM_ARGS}
time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 3 ${SIM_ARGS}

# our network
time python servoing_controller.py ${ARGS} --levels 0 1 2 3 --ladder 1 ${SIM_ARGS}
