#### aiap14-goh-jia-jun-745Z
Full name: Goh Jia Jun
email: gohjiajun.98@gmail.com

## Overview of submitted folders
```
aiap14-goh-jia-jun-745Z/
├── src/
│   ├── logs/
│   ├── models/
│   └── plots/
├── eda.ipynb 
├── model_config.yml
├── model_base_class.py
├── model.py
├── preprocess.py
├── utility.py
├── requirements.txt
└── run.sh
```

In the folder structure,
- logs/ represents the loggings of each model during training and evaluation
- model/ represents the model weights for the models chosen
- plot/ represents the Precision-Recall and ROC plot to evaluate each model's performance
- eda.ipynb contains the EDA for Task 1
- model_config.yml contains model parameters, folder paths and other necessary parameters
- model_base_class.py contains the abstract class for the models
- model.py contains the FishingModel class for training in Task 2 and its driver class
- preprocess.py contains preprocessing steps for feature processing
- utility.py contains common utility functions for processing, training and evaluation
- requirements.txt contains the dependencies
- run.sh contains the executable for the MLP pipeline

## Instructions
To run the whole pipeline execute the following 
1. `chmod +x run.sh` 
2. `./run.sh <model_number>`
- model_1 is RandomForest
- model_2 is XGBoost
- model_3 is K-Nearest Neigbours

To modify parameters, we refer to the model_config.yml. An example is shown in model_1 below

```
model_1:
  model_name: 'RandomForest'
  model_save_loc: models/rf_model.sav
  is_train: True
  save_model: True
  output_path: rf_prediction.csv
  log_path: logs/
  plot_path: plots/
  rf_n_est: [800, 1001, 100]
  rf_max_d: [6, 9, 1]
  threshold: 0.5
```
In this block of YAML code, we can change the parameters of RandomForest which are rf_n_est(representing number of estimators), rf_max_d(representing the max depth of the tress in the model) and threshold presenting the float where the model will consider it as a positive class "RainTomorrow"

## Logical steps/flow of the pipeline 
![Logical_steps_pipeline.jpg](attachment:Logical_steps_pipeline.jpg?raw=true "Optional Title")

## Overview of key findings from the EDA conducted in Task 1
- Choices done in pipeline

**Missing data**
 1. "None" value is renamed to "Unknown" in column "RainToday"

**Numerical columns**
   -  <b> Invalid data </b>
   1. Column "Sunshine" has negative values all are mutiplied by -1 assuming human error
   2. Removed value = 0 in column "Evaporation" has values containing 0

   - <b>Skewness</b>
   - Rainfall, WindGustSpeed, WindSpeed9am is Highly Skewed
   - Evaporation, Sunshine is Moderately Skewed
    1. I have used log(1+x) for the columns of 'WindGustSpeed', 'WindSpeed9am', 'Evaporation', 'WindSpeed3pm','Sunshine'as they were either Highly Skewed or Moderately Skewed.

   - <b>Kurtosis</b>
   - "Rainfall", "WindSpeed9am" has Heavy Tails
   - "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Cloud9am", "Cloud3pm", "AverageTemp" has Lighter Tails
      
**Categorical columns** 

   -  <b> Invalid data </b>
   - "Pressure9am" and "Pressure3pm" have unformatted columns. As such there is a need to change them to lower case values


   - <b>One-Hot-Encoding(OHC)</b>
   1. Required them to turn into continuous variables using one-hot-encoding as model is unable to read them

   - <b>BarPlot to measure Date</b>
   1. Shown to observe the frequency count of the Categorical columns
   2. Not much observable trend for year, month and day for RainTomorrow. Date seems to not be factor in "RainTomorrow"

**Responding variables**
   1. I have used SMOTE(Synthetic Minority Over-sampling Technique) to treat imbalance classes as there is 77.0% of "Yes" and 23.0% of "No"
   2. After conducting SMOTE, I have checked the Skewness and Kurtosis for all the variables
   3. Skewnes improved for almost all non-OHC apart from "Sunshine" and "Rainfall"
   4. OHC columns are naturally skewed as there are sparse

**Mulit-collinearity**
   - <b>Variance Influence Factors</b>
      1. From the Variance Influence Factors(VIF), if VIF > 10, there exisits multi-collinearity. If the model that we use depends on the linearity assumption, it will wise to remove those columns. However, for tree-based models, they might be more robust and columns can still be kept.
      
      
   - <b>Correlation matrix</b>
      1. From the correlation matrix, it is observed that is a somewhat high correlation(0.7 and above)
      - WindSpeed9am and WindGustSpeed
      - WindSpeed3pm and WindGustSpeed
      - Humidity3pm and Humidity9am
      - Cloud3pm and Cloud9am
      2. Again, if we use a linear model, it might be useful to remove WindGustSpeed, (Humidity3pm or Humidity 9am) , (Cloud3pm or Cloud9am)

## Table of how features were processed

| Column 1 Header | Column 2 Header | Column 3 Header | Column 4 Header | Column 5 Header |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| Row 1, Column 1 | Row 1, Column 2 | Row 1, Column 3 | Row 1, Column 4 | Row 1, Column 5 |
| Row 2, Column 1 | Row 2, Column 2 | Row 2, Column 3 | Row 2, Column 4 | Row 2, Column 5 |
| Row 3, Column 1 | Row 3, Column 2 | Row 3, Column 3 | Row 3, Column 4 | Row 3, Column 5 |


g. Explanation of your choice of models for each machine learning task.
## Explanation of choice of models

## Evaluation of the models developed and metrics used

Based on the logs provided, here are the model and their respective metrics after 5-fold cross validations with GridSearch


| Model          | Accuracy | Precision | AUC  | F1-Score | Sensitivity(Recall) | Specificity | Best Accuracy Threshold | Best F1 Threshold |
|----------------|----------|-----------|------|----------|---------------------|-------------|-------------------------|-------------------|
| KNN Training   | 0.87     | 0.80      | 0.86 | 0.88     | 0.98                | 0.75        | 0.6989                  | 0.6883            |
| KNN Testing    | 0.92     | 0.89      | 0.92 | 0.92     | 0.96                | 0.88        | 0.6989                  | 0.6883            |
| Random Forest (Training) | 0.86 | 0.85 | 0.86 | 0.87 | 0.88 | 0.84 | 0.4825 | 0.4825 |
| Random Forest (Testing) | 0.86 | 0.85 | 0.86 | 0.87 | 0.89 | 0.84 | 0.4825 | 0.4825 |
| XGBoost (Training) | 0.92 | 0.93 | 0.92 | 0.92 | 0.91 | 0.93 | 0.5122 | 0.5122 |
| XGBoost (Testing) | 0.92 | 0.94 | 0.92 | 0.92 | 0.91 | 0.94 | 0.5122 | 0.5122 |

From above, XGBoost seems to be best performing model as it has the highest metrics for for all training and tesing apart from Recall which KNN is higher.

For threshold determination, based on the Precision-Recall and ROC curve, it will be about roughly 0.51 to determine that a class is positive ("RainTomorrow")

## Consideration for deploying models developed


Consideration for deploying models developed
i. Other considerations for deploying the models developed.

NOTE: your EDA still deciding on the missing data part








