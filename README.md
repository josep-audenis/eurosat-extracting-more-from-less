# EuroSAT: Extracting More from Less

![Collage](/generated_assets/eurosat_collage.jpeg)

In this project I focus on how can feature extraction narrow the gap between classic **Machine Learning** algorithms and state-of-the-art **Convolutional Neural Networks** in the context of low resolution satellite images. The motivation behind this idea is to explore how environments with reduced capabilities (due to lack of budget or lower performing device hardware) can get resuslts nearly as good as high performance environments with GPUs and advanced CNNs. For this project I have chosen the EuroSAT dataset.

The EuroSAT dataset contains over 27.000 64x64 pixels images of the Earth surface taken by the ESA Sentinel-2 satellite, divided in 10 diferent categories. The official dataset works with 13 spectral bands, but for the purpose of the project I use a reduced [dataset](https://github.com/phelber/eurosat?tab=readme-ov-file) made of the same original images but with only 3 spectral bands.

I have modified the folder structure of the original dataset (erased the intermidiate folder containing the images, the relative path would be ``EuroSAT/{image_category}/{image_cataegory}_{i}.jpg``), in the repository you can find the dataset i have used. To track the dataset in your local repository, you might need Git [LFS](https://git-lfs.com/) due to the large size of it.