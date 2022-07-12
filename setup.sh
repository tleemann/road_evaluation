#!/bin/bash
conda create -n road
conda activate road
conda install -c pytorch python=3.8 torchvision=0.13 matplotlib=3.5 tqdm numpy=1.22 scipy=1.7 ipykernel # Base requirements
ipython kernel install --user --name=road # make kernel available to jupyter lab / notebook