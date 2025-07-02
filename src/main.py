import argparse
import sys

from models import train_classical
from features import generate_features_dataset

def main():
	parser = argparse.ArgumentParser(prog="EuroSAT classifier")
	parser.add_argument("-t", "--train", type=str, help="", choices=["classic", "neural network"], nargs=1)
	parser.add_argument("-e", "--extract", type=str, help="Specifiy the output filename (without extension), it will be stored in the /data/interim/ folder.", nargs=1)

	args = parser.parse_args()

	if args.train == ['classic']:
		train_classical()
	elif args.train == ["nerual network"]:
		print("train nn")
	elif len(args.extract) > 0:
		generate_features_dataset()
	else:	
		parser.print_usage()

if __name__ == "__main__":
	main()