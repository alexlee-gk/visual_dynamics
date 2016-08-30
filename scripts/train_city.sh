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
    echo "Usage: ./train_city.sh [X1_C_DIM [SHARE]]"
    exit
fi
echo "X1_C_DIM=${X1_C_DIM} SHARE=${SHARE}"

N=0
T=0
DATA=data/city32dof3aa_train_data_*.h5
VAL_DATA=data/city32dof3aa_val_data.h5
MAX_ITER=10000
ARGS="${DATA} --val_hdf5_fname ${VAL_DATA} --x1_c_dim ${X1_C_DIM} --lr 0.001 -p build_fcn_action_cond_encoder_net -t $T -n $N -v0 --max_iter ${MAX_ITER} --concat 1 --vis_rm 1000 -v1"

# ablated, one bilinear connection
#time python servoing_controller.py ${ARGS} --levels 3 --ladder 0 --no_train --no_sim --pf auto -v1
#time python servoing_controller.py ${ARGS} --levels 2 --ladder 0
#time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 --no_train --no_sim -v1
#time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 --postfix pretrained --pf models/theano/FcnActionCondEncoderNet_levels3_x1cdim32_numds0_bishare_ladder0_bn0_concat0_city32dof3aa_lr0.001/snapshot/_iter_10000.pkl #--no_train --no_sim -v1
#time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 --postfix pretrained --pf 36000 #--no_train --no_sim -v1
#time python servoing_controller.py ${ARGS} --levels 1 --ladder 0
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 &

#time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels3_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl &
#time python servoing_controller.py ${ARGS} --levels 2 3 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels23_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl &
#time python servoing_controller.py ${ARGS} --levels 1 2 3 --ladder 1  --pf models/theano/FcnActionCondEncoderNet_levels123_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl &
#time python servoing_controller.py ${ARGS} --levels 0 1 2 3 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels0123_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl &
#wait

#time python servoing_controller.py ${ARGS} --levels 0 1 2 3

# # same as above, but with ladder loss. initialize with the respective networks from above
# time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 --pf "models/caffe/FcnActionCondEncoderNet_levels3_x1cdim${X1_C_DIM}_numds0_share${SHARE}_ladder0_bn0_concat1_city32_lr0.001/snapshot/_iter_${MAX_ITER}.caffemodel" ${SIM_ARGS}
# time python servoing_controller.py ${ARGS} --levels 2 --ladder 1 --pf "models/caffe/FcnActionCondEncoderNet_levels2_x1cdim${X1_C_DIM}_numds0_share${SHARE}_ladder0_bn0_concat1_city32_lr0.001/snapshot/_iter_${MAX_ITER}.caffemodel" ${SIM_ARGS}
# time python servoing_controller.py ${ARGS} --levels 1 --ladder 1 --pf "models/caffe/FcnActionCondEncoderNet_levels1_x1cdim${X1_C_DIM}_numds0_share${SHARE}_ladder0_bn0_concat1_city32_lr0.001/snapshot/_iter_${MAX_ITER}.caffemodel" ${SIM_ARGS}

#time python servoing_controller.py ${ARGS} --levels 1 --ladder 0 && time python servoing_controller.py ${ARGS} --levels 1 --ladder 1 &





#time python servoing_controller.py ${ARGS} --levels 3 --ladder 0 &
#time python servoing_controller.py ${ARGS} --levels 2 --ladder 0 &
#time python servoing_controller.py ${ARGS} --levels 1 --ladder 0 &
#
## pixel-level at various resolutions. ladder loss is not applicable here
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 &
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 1 &
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 2 &
#time python servoing_controller.py ${ARGS} --levels 0 --ladder 0 --numds 3 &
#wait

#time python servoing_controller.py ${ARGS} --levels 3 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels3_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl &
#time python servoing_controller.py ${ARGS} --levels 2 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels2_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl &
#time python servoing_controller.py ${ARGS} --levels 1 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels1_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl &

# curriculum learning for our network. initialize with the network from the level above
#time python servoing_controller.py ${ARGS} --levels 2 3 --ladder 0 --pf models/theano/FcnActionCondEncoderNet_levels3_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl
##wait
#time python servoing_controller.py ${ARGS} --levels 1 2 3 --ladder 0 --pf models/theano/FcnActionCondEncoderNet_levels23_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl
#time python servoing_controller.py ${ARGS} --levels 0 1 2 3 --ladder 0 --pf models/theano/FcnActionCondEncoderNet_levels123_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl
#time python servoing_controller.py ${ARGS} --levels 0 1 2 3 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels0123_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl


#time python servoing_controller.py ${ARGS} --postfix curriculum --levels 2 3 --ladder 0 --pf models/theano/FcnActionCondEncoderNet_levels3_x1cdim${X1_C_DIM}_numds0_bishare_ladder1_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_${MAX_ITER}.pkl
#time python servoing_controller.py ${ARGS} --postfix curriculum --levels 2 3 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels23_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001_curriculum/snapshot/_iter_${MAX_ITER}.pkl
#time python servoing_controller.py ${ARGS} --postfix curriculum --levels 1 2 3 --ladder 0 --pf models/theano/FcnActionCondEncoderNet_levels23_x1cdim${X1_C_DIM}_numds0_bishare_ladder1_bn0_concat1_city32dof3aa_lr0.001_curriculum/snapshot/_iter_${MAX_ITER}.pkl
#time python servoing_controller.py ${ARGS} --postfix curriculum --levels 1 2 3 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels123_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001_curriculum/snapshot/_iter_${MAX_ITER}.pkl
#time python servoing_controller.py ${ARGS} --postfix curriculum --levels 0 1 2 3 --ladder 0 --pf models/theano/FcnActionCondEncoderNet_levels123_x1cdim${X1_C_DIM}_numds0_bishare_ladder1_bn0_concat1_city32dof3aa_lr0.001_curriculum/snapshot/_iter_${MAX_ITER}.pkl
#time python servoing_controller.py ${ARGS} --postfix curriculum --levels 0 1 2 3 --ladder 1 --pf models/theano/FcnActionCondEncoderNet_levels0123_x1cdim${X1_C_DIM}_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001_curriculum/snapshot/_iter_${MAX_ITER}.pkl

#time python servoing_controller.py ${ARGS} -p build_laplacian_fcn_action_cond_encoder_net --levels 0 1 2 3 --ladder 1
#time python servoing_controller.py ${ARGS} -p build_laplacian_fcn_action_cond_encoder_net --levels 0 1 2 3 --ladder 1 --pf models/theano/LaplacianFcnActionCondEncoderNet_levels0123_x1cdim32_numds0_bishare_ladder0_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_10000.pkl

#time python servoing_controller.py ${ARGS} -p build_fcn_action_cond_encoder_net --levels 3 --ladder 0 --pf auto --no_train --no_sim -v1 --vis_rm 1
#time python servoing_controller.py ${ARGS} -p build_fcn_action_cond_encoder_net --levels 3 --ladder 1 --pf auto --no_train --no_sim -v1 --vis_rm 1

#time python servoing_controller.py ${ARGS} -p build_fcn_action_cond_encoder_net --levels 3 --ladder 1 --postfix ladderall --pf models/theano/FcnActionCondEncoderNet_levels3_x1cdim32_numds0_bishare_ladder1_bn0_concat1_city32dof3aa_lr0.001/snapshot/_iter_10000.pkl
#time python servoing_controller.py ${ARGS} -p build_fcn_action_cond_encoder_net --levels 3 --ladder 1 --postfix ladderall --no_train --no_sim -v1 --vis_rm 1

python train.py config/predictor/vgg_levels012.yaml config/solver/bilinear_losslevels012_citydof3.yaml
