#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
export PYTHONPATH=../../
python NoRetrainingMethod.py \
		--data_path=/home/rong/roar_eval_yao/yao/data \
		--expl_path=/home/rong/roar_eval_yao/yao/data \
		--params_file='noretrain_params.json' \
		--model_path='../../data/cifar_8014.pth'
