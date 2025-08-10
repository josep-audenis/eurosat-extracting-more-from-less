import os
import shutil

DATASET_PATH = os.path.join(os.path.dirname(__file__), ".." , "data/external/EuroSAT.zip")
DESTINATION_PATH = os.path.join(os.path.dirname(__file__), "..",  "data/raw/")

def clean_directory(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == ".gitkeep":
                continue
            os.remove(os.path.join(root, file))

        for dir in dirs:
            full_dir = os.path.join(root, dir)
            shutil.rmtree(full_dir)


def unzip_dataset():
    os.system("unzip " + DATASET_PATH + " -d "  + DESTINATION_PATH)

if __name__ == "__main__":
    clean_directory(DESTINATION_PATH)
    unzip_dataset()