#### aiap14-goh-jia-jun-745Z
Full name: Goh Jia Jun
email: gohjiajun.98@gmail.com

b. Overview of the submitted folder and the folder structure.
## Overview of submitted folders
```
aiap14-goh-jia-jun-745Z/
├── src/
│   ├── logs/
│   ├── models/
│   └── plots/
├── eda.ipynb 
├── model_config.yml
├── model.py
├── preprocess.py
├── utility.py
├── requirements.txt
└── run.sh
```

c. Instructions for executing the pipeline and modifying any parameters.
## Instructions
To run the whole pipeline execute the following
- ./run.sh <model_number>
model_1 is RandomForest
model_2 is XGBoost
model_3 is K-Nearest Neigbours

## Logical steps/flow of the pipeline 
<Insert flowchart picture for pipeline>
d. Description of logical steps/flow of the pipeline. If you find it useful, please feel free to include suitable visualization aids (eg, flow charts) within the README.

## Overview of key findings from the EDA conducted in Task 1
- Choices done in pipeline
**Missing data**
 1. "None" value is renamed to "Unknown" in column "RainToday"

**Numerical columns**
   -  Invalid data
   1. Column "Sunshine" has negative values all are mutiplied by -1 assuming human error
   2. Removed value = 0 in column "Evaporation" has values containing 0
**Skewness and Kurtosis**
   - <b>Skewness</b>
   - Rainfall, WindGustSpeed, WindSpeed9am is Highly Skewed
   - Evaporation, Sunshine is Moderately Skewed
    1. I have used log(1+x) for the columns of 'WindGustSpeed', 'WindSpeed9am', 'Evaporation', 'WindSpeed3pm','Sunshine'as they were either Highly Skewed or Moderately Skewed.

   - <b>Kurtosis</b>
   - "Rainfall", "WindSpeed9am" has Heavy Tails
   - "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Cloud9am", "Cloud3pm", "AverageTemp" has Lighter Tails
      
**Categorical columns** 
   - "Pressure9am" and "Pressure3pm" have unformatted columns. As such there is a need to change them to lower case values

   - <b>One-Hot-Encoding(OHC)</b>
   1. Required them to turn into continuous variables using one-hot-encoding as model is unable to read them

   - <b>Barplot</b>
   1. Shown to observe the frequency count of the Categorical columns
   2. Not much observable trend for year, month and day for RainTomorrow. Date seems to not be factor in "RainTomorrow"

**Responding variables**
   1. I have used SMOTE(Synthetic Minority Over-sampling Technique) to treat imbalance classes as there is 77.0% of "Yes" and 23.0% of "No"
   2. After conducting SMOTE, I have checked the Skewness and Kurtosis for all the variables
   3. Skewnes improved for almost all non-OHC apart from "Sunshine" and "Rainfall"

**Mulit-collinearity**
   - Variance Influence Factors
      1. From the Variance Influence Factors(VIF), if VIF > 10, there exisits multi-collinearity. If the model that we use depends on the linearity assumption, it will wise to remove those columns. However, for tree-based models, they might be more robust and columns can still be kept.
   - Correlation matrix
      1. From the correlation matrix, it is observed that is a somewhat high correlation(0.7 and above)
      - WindSpeed9am and WindGustSpeed
      - WindSpeed3pm and WindGustSpeed
      - Humidity3pm and Humidity9am
      - Cloud3pm and Cloud9am
      2. Again, if we use a linear model, it might be useful to remove WindGustSpeed, (Humidity3pm or Humidity 9am) , (Cloud3pm or Cloud9am)

Summarized in table how features are processed
f. Described how the features in the dataset are processed (summarized in a table)

Explanation of choice of models
g. Explanation of your choice of models for each machine learning task.

h. Evaluation of the models developed. 
Any metrics used in the evaluation should also be explained.
Evaluation of models developed

Consideration for deploying models developed
i. Other considerations for deploying the models developed.

NOTE: your EDA still deciding on the missing data part (fuck it lol no time)








