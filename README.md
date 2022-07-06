# Remove And Debias (ROAD)
## An Efficient and Consistent Evaluation Strategy for Feature Attributions

The repository contains the source code of the (Remove And Debias) ROAD evaluation method for feature attribution methods. It is structured as follows:
* the folder ```road``` contains the interfaces and classes required to use the ROAD evaluation framework.
* the folder ```experiments``` contains code to replicate the results reported in our ICML paper.

## Paper
For a more profound introduction please have a look at our paper (available on arXiv for now).
Have a look at our paper:

Yao Rong, Tobias Leemann, Vadim Borisov, Gjergji Kasneci and Enkelejda Kasneci. ["A Constistent and Efficient Evaluation Strategy for Feature Attribution Methods"](https://arxiv.org/pdf/2202.00449), *International Conference on Machine Learning (ICML)*, PMLR, 2022


## Example
Check out the notebook ``RoadBenchmark.ipynb`` for an example.

Our benchmark can be included by using the files ``imputation.py`` and ``road.py``.

We run the script using:

-Python=3.8.8

-Pytorch=1.8.1

-torchvision=0.9

## Credits
Please cite us if you use our code or ressources in your own work.
