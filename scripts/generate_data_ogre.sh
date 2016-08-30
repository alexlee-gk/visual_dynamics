#!/bin/bash

python generate_data.py config/simulator/ogredof3.yaml -n2000 -t100 -o data/ogredof3_train_data
python generate_data.py config/simulator/ogredof3.yaml -n100 -t10 -o data/ogredof3_val_data