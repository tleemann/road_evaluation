#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
retrain=noretrain # noretrain
for impu in gan fixed linear 
do
for run in 0 1 2 3 4
do
echo $impu
{ time /bin/sh -c "export PYTHONPATH=../../; python3 runtime_comparison.py $impu $retrain" ; } 2>> file_${impu}_${retrain}.txt
done
done