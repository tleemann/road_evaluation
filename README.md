# Remove And Debias (ROAD)
## A Consistent and Efficient Evaluation Strategy for Feature Attributions

The repository contains the source code of the (Remove And Debias) ROAD evaluation method for feature attribution methods. It is structured as follows:
* the folder ```road``` contains the interfaces and classes required to use the ROAD evaluation framework.
* the folder ```experiments``` contains code to replicate the results reported in our ICML paper.

## Paper
For a more profound introduction please have a look at our paper (available on arXiv for now).
Have a look at our paper:

Yao Rong, Tobias Leemann, Vadim Borisov, Gjergji Kasneci and Enkelejda Kasneci. ["A Constistent and Efficient Evaluation Strategy for Feature Attribution Methods"](https://arxiv.org/pdf/2202.00449), *International Conference on Machine Learning (ICML)*, PMLR, 2022


## Example
Check out the notebook ``RoadBenchmarkDemo.ipynb`` for an example.

Our benchmark can be included in any project by adding the ``road`` module to the interpreter path. 
Subsequently, try running 

``from road import run_road``

and start evaluating faithfulness!

We run the script using:

-Python=3.8.8

-Pytorch=1.8.1

-torchvision=0.9

## Credits
Please cite us if you use our code or ressources in your own work, for instance with the following BibTex entry:
```
@inproceedings{rong2022evaluating,
  title={A Consistent And Efficient Evaluation Strategy for Feature Attribution Methods},
  booktitle={International Conference on Machine Learning},
  author={Rong, Yao and Leemann, Tobias and Borisov, Vadim and Kasneci, Gjergji and Kasneci, Enkelejda},
  year={2022},
  organization={PMLR}
}
```
