# MODEL_FILE="models/theano/multiscale_dilated_vgg_level5_scales012_infogain32/bilinear_losslevel5scales012_simplequaddof4largexgeomcar/_iter_1_model.yaml"
MODEL_FILE="models/theano/multiscale_levels012/bilinear_losslevels012_simplequaddof4largexgeomcar/_iter_1_model.yaml"
SCALE_CONDS="012 0 1 2"
# DISTANCE_CONDS="0 1 2 3 4 5"
# SCALE_CONDS="0"
DISTANCE_CONDS="3"
for D in ${DISTANCE_CONDS}; do
	for S in ${SCALE_CONDS}; do
		time python visual_servoing.py ${MODEL_FILE} -n100 -t100 -v0 -d ${D} -i ${S} -o results/scales${S}_d${D}
	done
done

# python visual_servoing.py models/theano/multiscale_dilated_vgg_level5_scales012_infogain32/bilinear_losslevel5scales012_simplequaddof4largexgeomcar/_iter_1_model.yaml -n100 -t100 -v0 -d3 --use_weights -o results/scales012_d3_weighted_vgg
# python visual_servoing.py models/theano/multiscale_dilated_vgg_level5_scales012_infogain32/bilinear_losslevel5scales012_simplequaddof4largexgeomcar/_iter_1_model.yaml -n100 -t100 -v0 -d3 --use_weights -o results/scales012_d3_weighted
