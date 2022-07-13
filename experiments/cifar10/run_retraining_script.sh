#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
export PYTHONPATH=../../
python RetrainingMethod.py \
		--data_path=/home/rong/roar_eval_yao/yao/data \
		--expl_path=/home/rong/roar_eval_yao/yao/data \
		--params_file='retrain_params.json'

