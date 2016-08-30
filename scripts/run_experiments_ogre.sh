#!/usr/bin/env bash

for FACTOR in $(seq 0 10); do
	python servoing_controller.py models/theano/multiscale_levels012/bilinear_losslevels012/_iter_1_model.yaml --dof_limit_factor $FACTOR -o results_ogre/multiscale_levels012_bilinear_losslevels012_$FACTOR -n100 -t100
done

for FACTOR in $(seq 0 10); do
	python servoing_controller.py models/theano/laplacian_levels012/bilinear_losslevels012_lossres/_iter_1_model.yaml --dof_limit_factor $FACTOR -o results_ogre/laplacian_levels012_bilinear_losslevels012_lossres_$FACTOR -n100 -t100
done

for FACTOR in $(seq 0 10); do
	python servoing_controller.py models/theano/multiscale_levels012/gamma0.9_losslevels012/_iter_100000_model.yaml --dof_limit_factor $FACTOR -o results_ogre/multiscale_levels012_gamma0.9_losslevels012_$FACTOR -n100 -t100
done

for FACTOR in $(seq 0 10); do
	python servoing_controller.py models/theano/laplacian_levels012/gamma0.9_losslevels012/_iter_100000_model.yaml --dof_limit_factor $FACTOR -o results_ogre/laplacian_levels012_gamma0.9_losslevels012_$FACTOR -n100 -t100
done

for FACTOR in $(seq 0 10); do
	python servoing_controller.py models/theano/multiscale_levels0/gamma0.9_losslevels012/_iter_100000_model.yaml --dof_limit_factor $FACTOR -o results_ogre/multiscale_levels0_gamma0.9_losslevels012_$FACTOR -n100 -t100
done

for FACTOR in $(seq 0 10); do
	python servoing_controller.py models/theano/laplacian_levels0/gamma0.9_losslevels012/_iter_100000_model.yaml --dof_limit_factor $FACTOR -o results_ogre/laplacian_levels0_gamma0.9_losslevels012_$FACTOR -n100 -t100
done

for FACTOR in $(seq 0 10); do
	python servoing_controller.py models/theano/multiscale_levels2/gamma0.9_losslevels012/_iter_100000_model.yaml --dof_limit_factor $FACTOR -o results_ogre/multiscale_levels2_gamma0.9_losslevels012_$FACTOR -n100 -t100
done

for FACTOR in $(seq 0 10); do
	python servoing_controller.py models/theano/laplacian_levels2/gamma0.9_losslevels012/_iter_100000_model.yaml --dof_limit_factor $FACTOR -o results_ogre/laplacian_levels2_gamma0.9_losslevels012_$FACTOR -n100 -t100
done


multiscale_levels012_gamma0.9_losslevels012

laplacian_levels012_bilinear_losslevels012_lossres
models/theano/multiscale_levels012/bilinear_losslevels012/_iter_1_model.yaml

subl models/theano/laplacian_levels012/bilinear_losslevels012_lossres/_iter_1_model.yaml

subl models/theano/multiscale_levels012/gamma0.9_losslevels012/_iter_100000_model.yaml

subl models/theano/laplacian_levels012/gamma0.9_losslevels012/_iter_100000_model.yaml


  class: !!python/name:simulator.OgreSimulator ''


for i in 0 1 2; do
    mkdir models/theano/laplacian_levels$i
    mkdir models/theano/laplacian_levels$i/gamma0.9_losslevels012
    cp models/theano/laplacian_levels012/gamma0.9_losslevels012/_iter_100000_model.yaml models/theano/laplacian_levels$i/gamma0.9_losslevels012/
    subl models/theano/laplacian_levels$i/gamma0.9_losslevels012/_iter_100000_model.yaml
done