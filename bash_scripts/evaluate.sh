#!/bin/bash

# Evaluate ECRTM on different datasets
echo "Evaluating ECRTM on 20NG dataset..."
python main.py --mode evaluate --path ./output/20NG/ECRTM_K50_1th_params.mat --label_path data/20NG --data_path ./output/20NG/ECRTM_K50_1th_T15

echo "Evaluating ECRTM on IMDB dataset..."
python main.py --mode evaluate --path ./output/IMDB/ECRTM_K50_1th_params.mat --label_path data/IMDB --data_path ./output/IMDB/ECRTM_K50_1th_T15

echo "Evaluating ECRTM on YahooAnswer dataset..."
python main.py --mode evaluate --path ./output/YahooAnswer/ECRTM_K50_1th_params.mat --label_path data/YahooAnswer --data_path ./output/YahooAnswer/ECRTM_K50_1th_T15

echo "Evaluating ECRTM on AGNews dataset..."
python main.py --mode evaluate --path ./output/AGNews/ECRTM_K50_1th_params.mat --label_path data/AGNews --data_path ./output/AGNews/ECRTM_K50_1th_T15

echo "All evaluations completed!"
