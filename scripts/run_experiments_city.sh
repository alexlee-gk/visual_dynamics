#!/bin/bash

if [ "$#" -eq 0 ]; then
	FACTOR=1
elif [ "$#" -eq 1 ]; then
	FACTOR=$1
else
    echo "Usage: ./run_experiments_city.sh [FACTOR]"
    exit
fi
echo "FACTOR=${FACTOR}"

X1_C_DIM=32
N=20
T=100
DATA=data/city32dof3aa_train_data_*.h5
VAL_DATA=data/city32dof3aa_val_data.h5
MAX_ITER=10000
ARGS="${DATA} --val_hdf5_fname ${VAL_DATA} --x1_c_dim ${X1_C_DIM} --lr 0.001 -p build_fcn_action_cond_encoder_net -t $T -n $N -v0 --max_iter ${MAX_ITER} --concat 1 --pf auto --no_train --experiment 0 --dof_limit_factor ${FACTOR} --output_results_dir results"

# ablated, one bilinear connection (with ladder loss)
#time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 &
#time python servoing_controller.py ${ARGS} --levels 2 --ladder 1 &
#time python servoing_controller.py ${ARGS} --levels 1 --ladder 1 &
##wait
##
### pixel-level at various resolutions. ladder loss is not applicable here
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 &
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 1 &
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 2 &
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 3 &
wait
#
## our network
time python servoing_controller.py ${ARGS} --levels 2 3 --ladder 1
#time python servoing_controller.py ${ARGS} --levels 1 2 3 --ladder 1
#time python servoing_controller.py ${ARGS} --levels 0 1 2 3 --ladder 1
