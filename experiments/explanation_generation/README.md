## Explanation Generation

To generate the explanations , please run the following command using this folder as cwd:
```python
sh expl_cifar.sh # on CIFAR-10 
```
or
```python
sh expl_food.sh # on Food-101
```

#### Preparation
+ Provide the path to the trained model that needs to be explained to  `--model_path`.

    A trained model on cifar can be found in the folder `road_evaluation/data/`. You can download the model trained on Food-101 [here](https://drive.google.com/file/d/19YCp30x8R_PRxjG4K4nIIE71HwBV7MR7/view?usp=sharing) and put it in the folder `road_evaluation/data/`.

+ Provide the path to the dataset to `--input_path` and the path to save the explanation `--save_path`.

+ Choose the explanation methods `--expl_method`: `ig`/ `gb` for IG / GB, `ig_sg`/ `gb_sg` for IG_SG / GB_SG,  `ig_sq`/ `gb_sq` for IG_SQ / GB_SQ,  `ig_var`/ `gb_var` for IG_Var / GB_Var

#### Example
A call example call to the script could look like this:
```
python3 ExplanationGeneration.py --expl_method=ig --model_path ../../data/cifar_8014.pth --save_path expl_save
```
More comments can be found in the scripts.
