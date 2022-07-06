# experiments/cifar10: "A Consistent and Efficient Evaluation Strategy for Feature Attribution Methods"

We run the script using:
-Python=3.8.8
-Pytorch=1.8.1
-torchvision=0.9

### Scripts:
1. "ExplanationGeneration.py": to generate different explanations (IG,GB-Families). For this, please install the package captum from here: https://captum.ai/

2. "RetrainingMethods.py": to run retraining methods,

   "NonRetrainingMethods.ipynb": to run non-retraining methods

3. "imputation_predict.py": Train a predictor to differentiate between imputed and original pixels. See "ImputationPrediction.ipynb" for additional 

4. "RankCorrelation.ipynb": to run the rank correlation analysis
5. "MaskLeakage.py": to run the experiments of class leakage through the mask


More instructions can be found in the scripts.

### Results:
All our results are in './data'.




