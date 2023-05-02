#!/bin/bash
# Install the required packages
pip install -r requirements.txt
cd src

cv=5
if [ ! -z "$2" ]; then
    cv="$2"
fi

# Train or predict using the specified model
if [ "$1" = "model_1" ]; then
    echo "Training model using ${cv}-fold cross validation using RandomForest"
    python model.py --model=model_1 --cv=$cv
elif [ "$1" = "model_2" ]; then
    echo "Training model using ${cv}-fold cross validation using XGBoost"
    python model.py --model=model_2 --cv=$cv
elif [ "$1" = "model_3" ]; then
    echo "Training model using ${cv}-fold cross validation using KNN"
    python model.py --model=model_3 --cv=$cv
else
    echo "Invalid model specified"
fi