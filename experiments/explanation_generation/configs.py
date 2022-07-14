import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    
    # Explanations params
    parser.add_argument("--gpu", type=bool,
                        help="use cuda")
    parser.add_argument("--batch_size", type=int,
                        help="batch size")
    parser.add_argument("--expl_method", type=str, 
                        help="explanation method: ig, gb, ig_sg, gb_sg, ig_sq, gb_sq, ig_var, gb_var")
    parser.add_argument("--input_path", type=str,
                        help="dataset input path")
    parser.add_argument("--save_path", type=str,
                        help="path to save the explanation")
    parser.add_argument("--model_path", type=str,
                        help="path to the trained model")
    parser.add_argument("--seed", type=int,
                        help="set random seed")


    parser.set_defaults(batch_size=32,
                        gpu=True,
                        expl_method='ig',
                        input_path='./data',
                        save_path='./data',
                        seed=42,
                        model_path='../../data/cifar_8014.pth'
                        )

    return parser.parse_args()

