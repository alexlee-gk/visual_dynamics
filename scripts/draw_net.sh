#!/bin/bash

if [ "$#" -eq 1 ]; then
	MODEL_DIR=$1
else
    echo "Usage: ./draw_net.sh MODEL_DIR"
    exit
fi
echo "MODEL_DIR=${MODEL_DIR}"

for TYPE in train_val deploy; do
	python ~/rll/caffe/python/draw_net.py ${MODEL_DIR}/${TYPE}.prototxt ${MODEL_DIR}/${TYPE}_net.pdf --rankdir BT
	evince ${MODEL_DIR}/${TYPE}_net.pdf&
done
