# EuroSAT: Extracting More from Less

![Collage](/docs/generated_assets/eurosat_collage.jpeg)

In this project I focus on how can feature extraction narrow the gap between classic **Machine Learning** algorithms and state-of-the-art **Convolutional Neural Networks** in the context of low resolution satellite images. The motivation behind this idea is to explore how environments with reduced capabilities (due to lack of budget or lower performing device hardware) can get results nearly as good as high performance environments with GPUs and advanced CNNs. For this project I have chosen the EuroSAT dataset.

The EuroSAT dataset contains over 27.000 64x64 pixels images of the Earth surface taken by the ESA Sentinel-2 satellite, divided in 10 different categories. The official dataset works with 13 spectral bands, but for the purpose of the project I use a reduced [dataset](https://github.com/phelber/eurosat?tab=readme-ov-file) made of the same original images but with only 3 spectral bands.

I have modified the folder structure of the original dataset (erased the intermediate folder containing the images, the relative path would be ``EuroSAT/{image_category}/{image_cataegory}_{i}.jpg``), in the repository you can find the dataset i have used. To track the dataset in your local repository, you might need Git [LFS](https://git-lfs.com/) due to the large size of it.

## Requirements

It's STRONGLY recommended to execute the project in Linux, the Python version used for the project is ``Python 3.11``. For installation of all dependencies in your environment execute the following command:

```bash
pip install -r requirements.txt
```

For automatic report compilation the project must be executed in a Linux environments with `texlive-latex-base` installed with:

```bash
apt install texlive-latex-base
```

As mentioned above the dataset is stored in this same repository using Git [LFS](https://git-lfs.com/), to verify that the file has been cloned correctly in your local repository is recommended to check the file format using the command ``file``. Once verified that the it has downloaded the Zip archive correctly, to extract all the content in the corresponding directory execute the ``prepare_dataset.py``script:

```bash
python src/prepare_dataset.py
```

If an EuorSAT dataset from another source is prefered, take into account that for executing further scripts and models the following dataset folder structure is expected:

```bash
eurosat-extracting-more-from-less/
├── data/
│   ├── external/
│   │   └── ...
│   ├── interim/
│   │   └── ...
│   └── raw/
│       └── EuroSAT/
│           ├── AnnualCrop/
│           │   ├── AnnualCrop_1.jpg
│           │   ├── AnnualCrop_2.jpg
│           │   └── ...
│           ├── Forest/
│           │   ├── Forest_1.jpg
│           │   ├── Forest_2.jpg
│           │   └── ...
│           ├── HerbaceousVegetation/
│           │   └── ...
│           └── ...
│
├── docs/
│   └── ...
└── ...
```

For **GPU** functionalities [NVIDIA cuda toolkit 12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive) was used. Take into account that ``cupy`` libary for gpu usage is dependent to de cuda toolkit. 

## Execution

For generating the feature dataset to train the classical models execute the ``exctract_features.py`` script using the following command:

```bash
python src/features/extract_features.py
```

This will generate a file named ``features.npz`` in the ``data/interim`` folder. If you want to generate different versions of a dataset please generate the dataset with different names, otherwise they will be overwritten.