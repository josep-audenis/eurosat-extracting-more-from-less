import argparse
import os

from models.train_classical_models import train_classic 
from features.extract_features import generate_features_dataset

def main():
    parser = argparse.ArgumentParser(prog="EuroSAT classifier")
    parser.add_argument("-t", "--train", type=str, help="Trains a classical model on the specified dataset.")
    parser.add_argument("-e", "--extract", type=str, help="Extracts specified featyres from the image dataset.")
    parser.add_argument("-c", "--check", type=str, help="Checks for dataset features statistics.")
    #pasrer.add_argument("-p", "--predict", type=str, help="Predict on a new image. Specify the path to the image.", nargs=1)
    parser.add_argument("-m", "--model", type=str, choices=["random_forest", "xgboost", "resnet50"], help="Specify the model to use for training or prediction.", nargs=1)
    parser.add_argument("-ld", "--load-dataset", type=str, help="Name of the dataset to load. It must be stored in the /data/interim/ folder.", nargs=1)
    parser.add_argument("-sd", "--save-dataset", type=str, help="Name of the dataset to be saved. If a split technic is applied the name may vary. It will be stored in the /data/interim/ folder.", nargs=1)
    parser.add_argument("-sm", "--save-model", type=str, help="Name to save the trained model. It will be stored in the /models/ folder.", nargs=1)
    parser.add_argument("-lm", "--load-model", type=str, help="Name of the trained model to load. It must be stored in the /models/ folder.", nargs=1)
    parser.add_argument("-r", "--report", action="store_true", help="Generates and saves a classification report.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize samples, predictions or feature distributions.")
    parser.add_argument("-eval", "--evaluate", type="str", help="Evaluates an existing trained model on a specified dataset.")
    parser.add_argument("-sr", "--split-ratio", type=int, help="Indicates the % of the dataset dedicated to training, the rest to test.", nargs=1)
    parser.add_argument("-cv", "--cross-validation", type=int, help="Indicates the number of folds for cross-validation", nargs=1)
    parser.add_argument("-s", "--seed", type=int, help="Seed to fix a random state.", nargs=1)

    args = parser.parse_args()

    if args.extract:
        if args.save_dataset:
            parser.error("--extract requires --save-dataset to be specified.")
        elif args.split_ratio and args.cross_validation:
            parser.error("Choose either --split-ratio or --cross-validation, not both.")
        else:
            print("extract")

    if args.train:
        if args.model and args.save_model and args.load_dataset:
            if args.model != "resnet50":
                print(f"train {model}")
                train_classic(args.train)
            else:
                parser.error("--train only supports classical models (random_forest, xgboost).")
        else:
            parser.error("--train requires --model, --save-model and --load-dataset to be specified.")

    if args.check:
        if args.load_dataset:
            parser.error("--check requires --load-dataset to be specified.")

    if args.evaluate:
        if args.load_model and args.load_dataset:
            print("evaluate")
        else:
            parser.error("--evaluate requires --load-model and --load-dataset to be specified.")

    #elif args.predict:
        #if args.load_model:
            #print(f"predict {args.load_model}")
        #else:
            #parser.error("--predict requires --load-model to be specified.")
    
    else:
        parser.print_usage()

if __name__ == "__main__":
    main()
