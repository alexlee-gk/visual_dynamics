#!/bin/bash

for i in $(seq 0 49); do
    time python generate_data.py -o original_data/servo_train_data_${i}.h5 -n100 -t100 servo --dof_min 250 280 --dof_max 590 390 --vel_min -25 -25 --vel_max 25 25
done
# use 2 and skip the next 3
python process_data.py original_data/servo_train_data_?([1-4])@([0-1]|[5-6]).h5 -o data/servo32_train_data_200k.h5
# use all 0-49
# python process_data.py original_data/servo_train_data_?([1-5])[0-9].h5 -o data/servo32_train_data_500k.h5

for i in 0; do
	time python generate_data.py -o original_data/servo_val_data_${i}.h5 -n100 -t100 servo --dof_min 250 280 --dof_max 590 390 --vel_min -25 -25 --vel_max 25 25
done
python process_data.py original_data/servo_val_data_0.h5 -o data/servo32_val_data_0.h5
