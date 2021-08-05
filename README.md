# Simple SVD with bias model for the Netflix Prize

This repository is a C implementation of the commonly used simple SVD with bias model for the Netflix Prize. If you are looking for a PyTorch implementation, please take a look at my blog post [here](https://wormtooth.com/20210805-netflix-svd/).

In order to run the model, you need to download the data from Kaggle: [Netflix Prize data](https://www.kaggle.com/netflix-inc/netflix-prize-data). After downloading the compressed data, you need to decompress the data into the same folder as the compiled program. The data folder name should be **data**:

```
data
├── README
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
├── movie_titles.csv
├── probe.txt
└── qualifying.txt
```
