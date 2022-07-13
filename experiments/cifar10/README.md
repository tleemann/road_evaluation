## Experiments on CIFAR10

### Scripts:

+ "run_retraining_script.sh": to run **Retrain** methods (ROAR). You can configure the training parameters in `retrain_params.json`. 
   "run_noretraining_script.sh": to run **No-Retrain** methods. You can configure the training parameters in `noretrain_params.json`. 
   ```python
   "basemethod": "ig" or "gb"
   "modifiers": ["base", "sg", "sq", "var"] # expl methods
   "imputation": "fixed" or "linear" or "gain" # "fixed": mean-value imputation, "linear": Noisy Linear Imputation, "gain": GAN imputation
    "morf": false or true  # true: morf, false: lerf 
    "datafile": "result/retrain.json"  # file to save the results
    "percentages": [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9] # eval percentages
    "timeoutdays": 0 # rerun the unfinished eval parameters n day before. 
   ```
      We provide two empty json files to store the retrain and no-retrain eval results. You can also generate these json files by runing:
   ```python  
   python utils.py --result_file='./result/filename.json'
   ```
   Please also change the path to the dataset and the saved explanations in the sh file.

+ "imputation_predict.py": Train a predictor to differentiate between imputed and original pixels. See "ImputationPrediction.ipynb" for additional 


More instructions can be found in the scripts.





