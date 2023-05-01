#!/bin/bash
# Install the required packages
pip install -r requirements.txt
cd src

if [ "$1" = "model_1" ]; then
    echo "Training/predicting using model1"
    python model.py --model=model_1
elif [ "$1" = "model_2" ]; then
    echo "Training/predicting using model2"
    python model.py --model=model_2
elif [ "$1" = "model_3" ]; then
    echo "Training/predicting using model3"
    python model.py --model=model_3
else
    echo "Invalid model specified"
fi
