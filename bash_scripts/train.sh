#!/bin/bash

# Run ECRTM on different datasets
echo "Running ECRTM on 20NG dataset..."
python main.py --mode train --model ECRTM --dataset 20NG --config src/configs/model/ECRTM_20NG.yaml --num_topic 50

echo "Running ECRTM on IMDB dataset..."
python main.py --mode train --model ECRTM --dataset IMDB --config src/configs/model/ECRTM_IMDB.yaml --num_topic 50

echo "Running ECRTM on YahooAnswer dataset..."
python main.py --mode train --model ECRTM --dataset YahooAnswer --config src/configs/model/ECRTM_YahooAnswer.yaml --num_topic 50

echo "Running ECRTM on AGNews dataset..."
python main.py --mode train --model ECRTM --dataset AGNews --config src/configs/model/ECRTM_AGNews.yaml --num_topic 50

echo "All training completed!"
