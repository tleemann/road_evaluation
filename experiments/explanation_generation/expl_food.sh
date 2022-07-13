#!/bin/bash
# Call this script from the experiments/explanation_generation folder as cwd.
python ExplanationGeneration_food.py \
		--input_path='/storage/rong/food-101' \
		--save_path='/storage/rong/food-101/expl/' \
		--model_path='../../data/plain_food.pth' \
		--batch_size=16 \
		--expl_method='ig'
