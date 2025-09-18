#!/bin/bash

# Evaluate ECRTM on different datasets
echo "Evaluating ECRTM on 20NG dataset..."
python src/utils/eva/TD.py --data_path ./output/20NG/ECRTM_K50_1th_T15
python src/utils/eva/cluster.py --path ./output/20NG/ECRTM_K50_1th_params.mat --label_path data/20NG

echo "Evaluating ECRTM on IMDB dataset..."
python src/utils/eva/TD.py --data_path ./output/IMDB/ECRTM_K50_1th_T15
python src/utils/eva/cluster.py --path ./output/IMDB/ECRTM_K50_1th_params.mat --label_path data/IMDB

echo "Evaluating ECRTM on YahooAnswer dataset..."
python src/utils/eva/TD.py --data_path ./output/YahooAnswer/ECRTM_K50_1th_T15
python src/utils/eva/cluster.py --path ./output/YahooAnswer/ECRTM_K50_1th_params.mat --label_path data/YahooAnswer

echo "Evaluating ECRTM on AGNews dataset..."
python src/utils/eva/TD.py --data_path ./output/AGNews/ECRTM_K50_1th_T15
python src/utils/eva/cluster.py --path ./output/AGNews/ECRTM_K50_1th_params.mat --label_path data/AGNews

echo "All evaluations completed!"
