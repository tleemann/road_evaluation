from datetime import datetime
import os, sys
from utils import append_evaluation_result, get_missing_run_parameters, update_eval_result, load_expl, arg_parse
import json
import time
import torchvision

## import from road module
import road
from road.imputations import *
from road.retraining import *


# different seeds
seeds = [2005, 42, 1515, 3333, 420]


if __name__ == '__main__':

    ## read configs
    args = arg_parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_path = args.data_path
    expl_path = args.expl_path
    use_device = torch.device("cuda" if args.gpu else "cpu")
    batch_size = args.batch_size
    params_file = args.params_file
    dataset = args.dataset

    ## set transforms
    transform_train = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #transforms.RandomHorizontalFlip(), 
    transform_test = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    params = json.load(open(params_file))
    print("Base Method Group: ", params["basemethod"])
    print("Types:", params["modifiers"])
    print("Imputation: ", params["imputation"])
    print("MoRF-order", bool(params["morf"]))
    print("Resultsfile",  params["datafile"])
    print("Percentages", params["percentages"])
    print("Timeout", int(params["timeoutdays"]))
    imputation = params["imputation"]
    group = params["basemethod"]
    morf = bool(params["morf"])
    storage_file = params["datafile"]
    modifiers = params["modifiers"]
    ps = params["percentages"]

    num_of_classes = 10  # 500
    if imputation == "linear":
        imputer = NoisyLinearImputer(noise=0.01)
    elif imputation == "fixed":
        imputer = ChannelMeanImputer()
    elif imputation == "gain":
        imputer = GAINImputer("../../road/gisp/models/cifar_10_best.pt", "cuda")

    ## set model
    model = models.resnet18

    run_params = get_missing_run_parameters(storage_file, imputation, morf, group, modifiers, ps, timeout=int(params["timeoutdays"]))
    print("Got Run Parameters (mod, perc, run_id): ", run_params)
    while run_params is not None:
        modifier = run_params[0]
        perc_value = run_params[1]
        run_id = run_params[2]
        torch.manual_seed(seeds[run_id]) # set appropriate seed 

        expl_train = f"{expl_path}/{group}/{modifier}_train.pkl"
        expl_test = f"{expl_path}/{group}/{modifier}_test.pkl"  

        start_time = time.time()
        ## load cifar 10 dataset in tensors
        transform_tensor = transforms.Compose([transforms.ToTensor()])
        cifar_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_tensor)
        cifar_test= torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_tensor)

        ## load explanation
        _, explanation_train, _, prediction_train = load_expl(None, expl_train)
        _, explanation_test, _, prediction_test = load_expl(None, expl_test)
    

        res_acc, prob_acc = retraining(dataset_train=cifar_train, dataset_test=cifar_test, transform_test=transform_test,
                                   explanations_train=explanation_train, explanations_test=explanation_test, transform_train=transform_train,
                                   predictions_train=prediction_train, predictions_test=prediction_test,
                                   num_of_classes=num_of_classes, modelclass=model, percentages=[perc_value], epoch=40, morf=morf, batch_size=batch_size, 
                                   save_path=storage_file, imputation=imputer)
        print('finished job with params', run_params, " Drawing new params.")
        print('--' * 50)
        print("--- %s seconds ---" % (time.time() - start_time))

        update_eval_result(res_acc[0].item(), storage_file, imputation, group, modifier, morf, perc_value, run_id)
        run_params = get_missing_run_parameters(storage_file, imputation, morf, group, modifiers, ps)
        print("Got Run Parameters (mod, perc, run_id): ", run_params)
        # exit()

    print("No more open runs. Terminiating.")
