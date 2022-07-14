import pickle 
import json
from datetime import datetime
import argparse

def load_dict(filename):
    with open(filename, 'rb') as f:
        di = pickle.load(f)
    return di

def load_expl(train_file, test_file):
    expl_train = []
    expl_test = []
    prediction_train = []
    prediction_test = []
    if train_file is not None:
        train_dict = load_dict(train_file)
        for i in range(len(train_dict)):
            # imshow_expl(train_dict[i]['expl'])
            expl_train.append(train_dict[i]['expl'])
            prediction_train.append(train_dict[i]['prediction'])
    if test_file is not None:
        test_dict = load_dict(test_file)
        for i in range(len(test_dict)):
            expl_test.append(test_dict[i]['expl'])
            prediction_test.append(test_dict[i]['prediction'])
    return expl_train, expl_test, prediction_train, prediction_test

def create_empty_evaluation_file(filepath, dataset):
    """ Create an empty JSON evaluation file where evaluation results can be stored in. """
    res_dict = {}
    res_dict["percentages"] = []
    res_dict["base_methods"] = []
    res_dict["modifiers"] = []
    res_dict["dataset"] = dataset
    res_dict["imputations"] = []
    res_dict["orders"] = []
    json.dump(res_dict, open(filepath, "w"))
    

def append_evaluation_result(value: float, filepath: str, imputation: str, base_method: str,
        modifier: str, use_morf: bool, percentage_value: float, only_new_idx=-1):
    """ Append an evaluation result to an existing JSON file.
        only_new_idx: only append this result, if the number of results present is smaller than only_new_idx
            -1 means always append.
        return True if the results were appended.
    """
    res_dict = json.load(open(filepath))
    if imputation not in res_dict["imputations"]:
        res_dict["imputations"].append(imputation)
        res_dict[imputation] = {}

    target_dict = res_dict[imputation]
    
    if base_method not in res_dict["base_methods"]:
        res_dict["base_methods"].append(base_method)
    if base_method not in target_dict.keys():
        target_dict[base_method] = {}
    target_dict = target_dict[base_method]

    if modifier not in res_dict["modifiers"]:
        res_dict["modifiers"].append(modifier)
    if modifier not in target_dict.keys():
        target_dict[modifier] = {}
    target_dict = target_dict[modifier]

    morfstr = "morf" if use_morf else "lerf"
    if morfstr not in res_dict["orders"]:
        res_dict["orders"].append(morfstr)
    if morfstr not in target_dict.keys():
        target_dict[morfstr] = {}
    target_dict = target_dict[morfstr]

    if percentage_value not in res_dict["percentages"]:
        res_dict["percentages"].append(percentage_value)
    if str(percentage_value) not in target_dict.keys():
        target_dict[str(percentage_value)] = [] 
    target_list = target_dict[str(percentage_value)]
    if only_new_idx == -1 or len(target_list) <= only_new_idx:
        target_list.append(value)
        json.dump(res_dict, open(filepath, "w"))
        return True
    return False

def getresultslist(filepath: str, imputation: str, base_method: str,
        modifier: str, use_morf: bool, percentage_value: float):
    """ Append an evaluation result to an existing JSON file.
        only_new_idx: only append this result, if the number of results present is smaller than only_new_idx.
            -1 means always append.
        return True if the results were appended.
    """
    res_dict = json.load(open(filepath))
    if imputation not in res_dict["imputations"]:
        res_dict["imputations"].append(imputation)
        res_dict[imputation] = {}

    target_dict = res_dict[imputation]
    
    if base_method not in res_dict["base_methods"]:
        res_dict["base_methods"].append(base_method)
    if base_method not in target_dict.keys():
        target_dict[base_method] = {}
    target_dict = target_dict[base_method]

    if modifier not in res_dict["modifiers"]:
        res_dict["modifiers"].append(modifier)
    if modifier not in target_dict.keys():
        target_dict[modifier] = {}
    target_dict = target_dict[modifier]

    morfstr = "morf" if use_morf else "lerf"
    if morfstr not in res_dict["orders"]:
        res_dict["orders"].append(morfstr)
    if morfstr not in target_dict.keys():
        target_dict[morfstr] = {}
    target_dict = target_dict[morfstr]

    if percentage_value not in res_dict["percentages"]:
        res_dict["percentages"].append(percentage_value)
    if str(percentage_value) not in target_dict.keys():
        target_dict[str(percentage_value)] = [] 
    target_list = target_dict[str(percentage_value)]
    return res_dict, target_list

def merge_new_results(file_old, file_new):
    res_dict2 = json.load(open(file_new))
    for im in res_dict2["imputations"]:
        res = res_dict2[im]
        for bm in res.keys():
            res2 = res[bm]
            for mod in res2.keys():
                res3 = res2[mod]
                for morflerf in res3.keys():
                    res4 = res3[morflerf]
                    for p in res4.keys():
                        res5 = res4[p]
                        for i, k in enumerate(res5):
                            _, list = getresultslist(file_old, im, bm, mod, morflerf=="morf", float(p))
                            if len(list) <= i:
                                append_evaluation_result(res5[i], file_old, im, bm, mod, morflerf=="morf", float(p), only_new_idx=i)
                            elif str(list[i])[:7] == "pending" and str(k)[:7] != "pending":
                                # Update pending runs
                                print("Updating pending run...")
                                update_eval_result(res5[i], file_old, im, bm, mod, morflerf=="morf", float(p), i)


def update_eval_result(value: float, filepath: str, imputation: str, base_method: str, modifier: str, use_morf: bool, percentage_value: float, run_id: int):
    res_dict, mylist = getresultslist(filepath, imputation, base_method, modifier, use_morf, percentage_value)
    mylist[run_id] = value
    json.dump(res_dict, open(filepath, "w"))

def get_missing_run_parameters(filepath, imputation, order: bool, base_method, modifiers, percentages, target_num_runs=5, timeout=3):
    """ Return a tuple of percentage value, runID, that still needs to be computed. """
    for mod in modifiers:
        for p in percentages:
            res_dict, mylist = getresultslist(filepath, imputation, base_method, mod, order, p)
            # Clean outdated runs
            for k in range(len(mylist)):
                if str(mylist[k])[:7] == "pending":
                    last_date = datetime.strptime(mylist[k][7:], "%d-%m-%Y,%H:%M:%S")
                    if (datetime.now()-last_date).days >= timeout:
                        print("Found outdated pending run...")
                        mylist[k] = "pending"+datetime.now().strftime("%d-%m-%Y,%H:%M:%S")
                        json.dump(res_dict, open(filepath, "w"))
                        return (mod, p, k)
            if len(mylist) < target_num_runs:
                mylist.append("pending"+datetime.now().strftime("%d-%m-%Y,%H:%M:%S"))
                json.dump(res_dict, open(filepath, "w"))
                return (mod, p, len(mylist)-1)
    return None


def arg_parse():
    parser = argparse.ArgumentParser()
    
    # Explanations params
    parser.add_argument("--gpu", type=bool,
                        help="use cuda")
    parser.add_argument("--batch_size", type=int,
                        help="batch size")
    parser.add_argument("--expl_path", type=str, 
                        help="path to saved explanations")
    parser.add_argument("--data_path", type=str,
                        help="dataset input path")
    parser.add_argument("--model_path", type=str,
                        help="path to the trained model")
    parser.add_argument("--seed", type=int,
                        help="set random seed")
    parser.add_argument("--result_file", type=str,
                        help="filename of the json file to save the results")
    parser.add_argument("--dataset", type=str,
                        help="the name of the dataset. E.g., cifar10")
    parser.add_argument("--params_file", type=str,
                        help="path of the json file of parameters")    


    parser.set_defaults(batch_size=32,
                        gpu=True,
                        expl_path='../explanation_generation/data',
                        input_path='../explanation_generation/data',
                        seed=42,
                        model_path='../../data/cifar_8014.pth',
                        result_file='./result/retrain.json',
                        dataset='cifar10',
                        params_file='retrain_params.json'
                        )

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    print("Generate a result json file at %s for the dataset %s."%(args.result_file, args.dataset))
    create_empty_evaluation_file(args.result_file, args.dataset)
