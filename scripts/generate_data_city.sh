#!/bin/bash

for i in $(seq 0 4); do
    python generate_data.py -o original_data/city_train_data_${i}.h5 -n800 -t50 city
done
python process_data.py original_data/city_train_data_[0-4].h5 -o data/city32_train_data_0-4.h5

python generate_data.py -o original_data/city_val_data.h5 -n100 -t10 city
python process_data.py original_data/city_val_data.h5 -o data/city32_val_data.h5
