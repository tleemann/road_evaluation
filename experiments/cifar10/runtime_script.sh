#!/bin/bash
# Call me with two parameters. runtime_script <imputation> <retrain>
retrain=retrain
for impu in fixed linear gan
do
for run in 0 1 2 3 4
do
echo $impu
{ time /bin/sh -c "export PYTHONPATH=.; python3 cifar10/runtime_comparison.py $impu $retrain" ; } 2>> file_${impu}_${retrain}.txt
done
done