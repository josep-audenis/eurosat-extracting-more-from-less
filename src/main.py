import argparse

from models.train_classical_models import train_classic 
from features.extract_features import generate_features_dataset

def main():
	parser = argparse.ArgumentParser(prog="EuroSAT classifier")
	parser.add_argument("-t", "--train", type=str, help="", choices=["classic", "neural network"], nargs=1)
	parser.add_argument("-e", "--extract", type=str, help="Specifiy the output filename (without extension), it will be stored in the /data/interim/ folder.", nargs=1)

	args = parser.parse_args()

	if args.extract:
		generate_features_dataset()
	elif args.train == ['classic']:
		train_classic()
	elif args.train == ["nerual network"]:
		print("train nn")
	else:
		parser.print_usage()

if __name__ == "__main__":
	main()