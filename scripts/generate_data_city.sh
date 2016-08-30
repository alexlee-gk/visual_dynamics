#!/bin/bash

# for i in $(seq 0 4); do
#     python generate_data.py -o original_data/city_train_data_${i}.h5 -n800 -t50 city
# done
# python process_data.py original_data/city_train_data_[0-4].h5 -o data/city32_train_data_0-4.h5

# python generate_data.py -o original_data/city_val_data.h5 -n100 -t10 city
# python process_data.py original_data/city_val_data.h5 -o data/city32_val_data.h5


# for i in $(seq 0 4); do
#     python process_data.py original_data/citydof3_train_data_${i}.h5 -o data/city32dof3_train_data_${i}.h5 &
# done

# python process_data.py original_data/city_val_data.h5 -o data/city84_val_data.h5 --image_scale 0.175 --crop_size 84 84
# for i in $(seq 0 4); do
# 	python process_data.py original_data/city_train_data_${i}.h5 -o data/city84_train_data_${i}.h5 --image_scale 0.175 --crop_size 84 84
# done


#for i in $(seq 0 4); do
#    # python generate_data.py -o original_data/citydof3aa_train_data_${i}.h5 -n800 -t50 city --dof_min -110 70 0 --dof_max -80 110 75 --dof 3;
#    python process_data.py original_data/citydof3aa_train_data_${i}.h5 -o data/city32dof3aa_train_data_${i}.h5 &
#done
# python generate_data.py -o original_data/citydof3aa_val_data.h5 -n800 -t50 city --dof_min -110 70 0 --dof_max -80 110 75 --dof 3;
#python process_data.py original_data/citydof3aa_val_data.h5 -o data/city32dof3aa_val_data.h5 &

#python generate_data.py -o original_data/citydof3aa_val_data_1.h5 -n100 -t10 city --dof_min 200 70 10 --dof_max 230 110 55 --dof 3;
#python process_data.py original_data/citydof3aa_val_data_1.h5 -o data/city32dof3aa_val_data_1.h5 &
#wait

#python generate_data.py -o test_original_data/citydof3aa_val_data.h5 -n10 -t10 city;

python generate_data.py config/simulator/citydof3.yaml -n 100 -t 10 -o data/citydof3_val_data
python generate_data.py config/simulator/citydof3.yaml -n 2000 -t 100 -o data/citydof3_train_data
