#!/bin/bash

# Preprocess 20NG dataset
echo "Preprocessing 20NG dataset..."
python src/preprocess/download_20ng.py
python src/preprocess/preprocess.py -d data/raw_data/20ng/20ng_all --output_dir data/20NG --vocab-size 5000 --label group

# Preprocess IMDB dataset
echo "Preprocessing IMDB dataset..."
python src/preprocess/download_imdb.py
python src/preprocess/preprocess.py -d data/raw_data/imdb --output_dir data/IMDB --vocab-size 5000 --label sentiment

echo "All preprocessing completed!"
