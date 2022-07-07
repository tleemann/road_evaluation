

## Source Code for: Generative Imputation and Stochastic Prediction


### Citation
```
M. Kachuee, K. Kärkkäinen, O. Goldstein, S. Darabi, M. Sarrafzadeh, 
Generative Imputation and Stochastic Prediction, 
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.
```

### Project Structure
- **train.py**: main entry point of the program, used to run different experiments
- **utils.py**: a set of utility functions
- **test_cifar.sh**: a simple script to run GI experiment for the CIFAR-10 dataset
- **README.md**: this readme file
- **environment.py**: conda environment settings
- **data.py**: load and preprocessing for different datasets
- **otherwork/**: source code from other work (with some modifications to integrate)
- **models/**: pytorch model architectures and modules
- **imputers/**: training process for different imputers

`Note: as shown in the example below, always fix the hash seed. We use python hashing to fingerprint samples.`

### Command example
```
export PYTHONHASHSEED=0
python3 train.py --exp "ENS_EPS2000" --dataset cifar10 --data_dir ~/Database/Image/ \
   --objective bce --lr_d 0.0005 --lr_g 0.0005 --lr_patience 0.25 \
   --missing_type mcar_rect --missing_rate 0.20 --hint_rate 0.0 --alpha 0.0 \
   --device cuda:0 --epoches 2000 --eval_freq 0.05 --batch_size 64 \
   --train_predictor --n_samples 128  --aug_noise_std 0.0 \
   --result_dir ./run_outputs/ --dump_ens
```
