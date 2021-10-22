#!/bin/bash

# bash script to run model training 5 times
# before running the script set use_id in the configuration file to run the experiment with the same name 5 times
for i in {1..5}; do CUDA_VISIBLE_DEVICES=7 python example.py -l labeled_anomalies.csv; done