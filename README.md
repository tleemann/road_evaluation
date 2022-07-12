# Remove And Debias (ROAD)
## A Consistent and Efficient Evaluation Strategy for Feature Attributions

<img align="right" width="300" height="240" src="https://raw.githubusercontent.com/tleemann/road_evaluation/road_v2/images/ICMLPaperTeaser.png">

The repository contains the source code of the **Remove And Debias (ROAD)** evaluation method for feature attribution methods. 

Unlike other evaluation frameworks, ROAD prevents *Class Information Leakage*, a phenomenon that may distort the evaluation results. This results in a more constistent benchmark, even with the different removal orders Most Relevant First (MoRF) and Least Relevant First (LeRF). Furthermore, it is computationally efficient and requires no costly model retraining steps (see Figure).

This repository is structured as follows:
* the folder ```road``` contains the Python package with the interfaces and classes required to use the ROAD evaluation framework.
* the folder ```experiments``` contains code to replicate the results reported in our ICML paper.
* we provide pretrained classification and [GAIN imputation](https://proceedings.mlr.press/v80/yoon18a.html) models for the CIFAR-10 dataset, as well as [Integrated Gradients](https://arxiv.org/abs/1703.01365) (IG) attributions in this repository to allow a quick start with the benchmark.

## Overview
<img align="left" width="350" height="350" src="https://raw.githubusercontent.com/tleemann/road_evaluation/road_v2/images/imputation_cifar.png">

Attribution methods are explainability techniques, that assign importance scores to input features (i.e., pixels in a computer vision context). With many attribution methods being proposed in the recent literature, the need for sound strategies to evaluate these attribution methods arises. A key idea is to remove the pixels considered most relevant by an attribution for the data samples and report the drop in accuracy. But how does one remove a pixel without destroying the entire image? The validity of the benchmark is determined by the implementation of the removal routine. ROAD introduces a *Noisy Linear Imputation* operator that is simple to implement and keeps the dependencies intact while provably removing the information contained in the chosen pixels (see Figure on the left).
<br>
<br>
<br>
<br>

## Paper
For a more profound introduction please have a look at our paper (available on arXiv for now).

Yao Rong, Tobias Leemann, Vadim Borisov, Gjergji Kasneci and Enkelejda Kasneci. ["A Constistent and Efficient Evaluation Strategy for Attribution Methods"](https://arxiv.org/pdf/2202.00449), *International Conference on Machine Learning (ICML)*, PMLR, 2022


## Getting started
### Conda environment
We recommend setting up an extra conda environment for this code to ensure matching versions of the dependencies are installed. To setup the environment and run the notebooks, we assume you have a working installation of Anaconda and Jupyter that and your shell is correctly configured to use the ``conda`` and ``jupyter`` commands.

In this case, you can setup the environment and the corresponding Jupyter Kernel by running the install scripts corresponding to your OS on the terminal:

**Linux**
```
source setup.sh
```
**Windows PowerShell**
```
./setup.ps1
```
Don't forget to answer YES, when promted.
You can now use your existing installation of Jupyter Notebook / Lab with the ``road``-kernel (don't forget to restart the Jupyter Server to see the kernel). This should allow you to run the scripts and the notebooks in this repository. 

### Tutorial
We recommend to take a look at the notebook ``RoadBenchmarkDemo.ipynb`` first where the most relevant features are explained.

Our benchmark can be included in any project by adding the ``road`` module to the interpreter path. 
Subsequently, try running 

```python
from road import run_road
```

and start evaluating faithfulness!

## Credits
Please cite us if you use our code or ressources in your own work, for instance with the following BibTex entry:
```
@InProceedings{rong22consistent,
  title     = {A Consistent and Efficient Evaluation Strategy for Attribution Methods},
  author    = {Rong, Yao and Leemann, Tobias and Borisov, Vadim and Kasneci, Gjergji and Kasneci, Enkelejda},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  pages     = {18770--18795},
  year      = {2022},
  publisher = {PMLR}
}

```
