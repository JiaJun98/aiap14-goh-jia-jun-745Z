import pandas as pd
import os
import numpy as np
import argparse
import time
from tqdm import tqdm
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from model_base_class import BaseModel
from utility import custom_print, churn_eval_metrics, plot_pr_curve, parse_config, plot_roc_curve
from preprocess import load_dataset, train_test_processing, preprocess_data
from xgboost import XGBClassifier
import pickle
import joblib

#Need to optimise
#Need to provide explanation for choice of algorithms
#Appropriate use of eval metrics + evaluation (Follow DSA4263)
#Understand diff components in ML pipelines

#Create Non-Bert Classifier Class
class FishingModel(BaseModel):
    '''
    FishingModel
    Parameters
    -----------
    model_name: str
        FishingModel Model to train/predict. Choose from 'LogisticRegression', 'RandomForest', 'XGBoost'

    Attributes
    -----------
    model: str
        The type of model that will be used in training/prediction of the data
        ('LogisticRegression', 'RandomForest', 'XGBoost')
    data: pd.DataFrame
       Fully cleaned DataFrame
    x_train: pd.DataFrame
        Contains preprocessed bag of words features that will be used in training
    x_test: pd.DataFrame
        Contains preprocessed bag of words features that will be used in testing trained model
    y_train: np.ndarray
        True value of 'RainTomorrow' to be used in training
    y_test: np.ndarray
        True value of 'RainTomorrow' to be used to obtain evaluation metrics
    
    Methods
    -----------
    predict(model_type, threshold):
        Predicts the given input data using the specified model_type and
        outputs the predictions (containing 0's or 1's) into a csv file.
    train(model_type):
        Trains the chosen model using the respective data given
    logreg():
        When 'LogisticRegression' is chosen for parameter model_type in the train function,
        this function will be called to train a Logistic Regression model
    rf():
        When 'RandomForest' is chosen for parameter model_type in the train function,
        this function will be called to train a Random Forest model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are n_estimators and max_depth
    xgboost():
        When 'XGBoost' is chosen for parameter model_type in the train function,
        this function will be called to train a XGBoost model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are eta, max_depth, min_child_weight,
        n_estimators and colsample_bytree
    '''
    def  __init__(self, x_train, x_test, y_train, y_test, model_name = None):
        self.model = model_name
        self.x_train = x_train
        self.x_test =  x_test
        self.y_train = y_train
        self.y_test =  y_test

    def train(self, model_type):
        '''
        Trains the chosen model using the respective data given
        
        Parameters
        -----------
            model_type : str
                A choice of 3 models are available.
                'RandomForest' and 'XGBoost'
        '''
        if model_type == 'RandomForest':
            self.rf()

        if model_type == 'XGBoost':
            self.xgboost()
        if model_type == 'KNN':
            self.knn()
    
    def predict(self, model_type, threshold):
        '''
        Predicts the given input data using the specified model_type
        and outputs the predictions (containing 0's or 1's) into a csv file.
        
        Parameters
        -----------
            model_type: str
                A choice of 3 models are available. 'LogisticRegression',
                'RandomForest' and 'XGBoost'
            threshold: float
                Threshold to decide at what probability the sentiment would be considered positive
        '''
        if model_type == 'RandomForest':
            rf = pickle.load(open(model_save_loc, 'rb'))
            sentiment_proba = rf.predict_proba(self.data)[:,1]
            sentiment = []
            for i in sentiment_proba:
                if i >= threshold:
                    sentiment.append(1)
                else:
                    sentiment.append(0)

        if model_type == 'XGBoost':
            xgb = pickle.load(open(model_save_loc, 'rb'))
            sentiment_proba = xgb.predict_proba(self.data)[:,1]
            sentiment = []
            for i in sentiment_proba:
                if i >= threshold:
                    sentiment.append(1)
                else:
                    sentiment.append(0)
        if model_type == 'Knn':
            knn = pickle.load(open(model_save_loc, 'rb'))
            sentiment_proba = knn.predict_proba(self.data)[:,1]
            sentiment = []
            for i in sentiment_proba:
                if i >= threshold:
                    sentiment.append(1)
                else:
                    sentiment.append(0)
        custom_print(str(model_type) + ' has been succesfully predicted\n', logger = logger)

    def rf(self): #Currently testing CV = 2
        '''
        When 'RandomForest' is chosen for parameter model_type in the train function,
        this function will be called to train a Random Forest model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are n_estimators and max_depth
        '''
        #Load training parameter range
        n_est_range = config_file[model_number]['rf_n_est']
        n_est = np.arange(n_est_range[0], n_est_range[1], n_est_range[2])
        max_d_range = config_file[model_number]['rf_max_d']
        max_d = np.arange(max_d_range[0], max_d_range[1], max_d_range[2])
        rf_grid = {'max_depth':max_d, 'n_estimators':n_est}
        #Execute grid search and fit model
        rf = RandomForestClassifier(random_state = 4263, criterion = 'entropy')
        rf_gscv = GridSearchCV(rf, rf_grid, cv = 2, return_train_score = True, verbose=10)
        rf_gscv.fit(self.x_train, self.y_train)
        start_time = time.time()
        rf_para = rf_gscv.best_params_
        
        rf = RandomForestClassifier(n_estimators = rf_para.get('n_estimators'),
                                    max_depth = rf_para.get('max_depth'),
                                    criterion = 'entropy', random_state = 4263)
        #sys.stdout=open("external_file.txt","w")
        rf.fit(self.x_train, self.y_train)
        finished = time.time()
        time_elaspsed = finished - start_time
        hours, seconds = divmod(time_elaspsed, 3600)
        minutes, seconds = divmod(seconds, 60)
        custom_print("Current training time:",logger = logger)
        custom_print("{:02d}:{:02d}:{:06.2f}".format(int(hours), int(minutes), seconds),
                     logger = logger)
        rf_pred = rf.predict(self.x_test)
        rf_proba = rf.predict_proba(self.x_test)[:,1]
        custom_print(rf_gscv.best_params_, logger = logger)
        custom_print('RandomForest model succesfully trained\n', logger = logger)
        custom_print('---------------------------------\n',logger = logger)
        churn_eval_metrics(rf_pred, self.y_test, logger)
        custom_print('\n---------------------------------\n',logger = logger)
        custom_print('Threshold parameter tuning\n', logger = logger)
        threshold, accuracy = plot_pr_curve(rf_proba, self.y_test, plot_path)
        plot_roc_curve(rf_proba, self.y_test, plot_path)
        rf_pred_best = []
        for i in rf_proba:
            if i>=threshold:
                rf_pred_best.append(1)
            else:
                rf_pred_best.append(0)
        custom_print('Prediction using best threshold for accuracy\n-------------------------\n',
                             logger = logger)                
        churn_eval_metrics(rf_pred_best, self.y_test, logger)
        custom_print('Best threshold for accuracy: ' + str(threshold), logger = logger)
        custom_print('Accuracy score at best threshold: ' + str(accuracy), logger = logger)
        if save_model:
            pickle.dump(rf, open(model_save_loc, 'wb'))
            custom_print('RandomForest model succesfully saved', logger = logger)
        else:
            custom_print('Warning: RandomForest model has NOT been saved', logger = logger)

    def xgboost(self):
        '''
        When 'XGBoost' is chosen for parameter model_type in the train function,
        this function will be called to train a XGBoost model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are eta, max_depth, min_child_weight,
        n_estimators and colsample_bytree
        '''
        #Loading training parameter range
        eta_range = config_file[model_number]['xgb_eta']
        eta = np.arange(eta_range[0], eta_range[1], eta_range[2])
        max_d_range = config_file[model_number]['xgb_max_d']
        max_d = np.arange(max_d_range[0], max_d_range[1], max_d_range[2])
        min_weight_range = config_file[model_number]['xgb_min_weight']
        min_weight = np.arange(min_weight_range[0], min_weight_range[1], min_weight_range[2])
        n_est_range = config_file[model_number]['xgb_n_est']
        n_est = np.arange(n_est_range[0], n_est_range[1], n_est_range[2])
        sample_range = config_file[model_number]['xgb_sample']
        sample = np.arange(sample_range[0], sample_range[1], sample_range[2])
        xgb_grid = {'eta':eta, 'max_depth':max_d, 'min_child_weight':min_weight,
                    'colsample_bytree':sample, 'n_estimators':n_est}
        #Execute grid search and fit model
        xgb = XGBClassifier(random_state = 4263, eval_metric = roc_auc_score)
        xgb_gscv = GridSearchCV(xgb, xgb_grid, cv = 2 ,return_train_score = True, verbose=10)
        xgb_gscv.fit(self.x_train, self.y_train)
        xgb_para = xgb_gscv.best_params_
        xgb = XGBClassifier(eta = xgb_para.get('eta'), max_depth = xgb_para.get('max_depth'),
                            min_child_weight = xgb_para.get('min_child_weight'),
                            colsample_bytree = xgb_para.get('colsample_bytree'),
                            n_estimators = xgb_para.get('n_estimators'), random_state = 4263,
                            eval_metric = roc_auc_score)
        xgb.fit(self.x_train, self.y_train)
        xgb_pred = xgb.predict(self.x_test)
        xgb_proba = xgb.predict_proba(self.x_test)[:,1]
        custom_print('XGBoost model succesfully trained\n', logger = logger)
        custom_print(xgb_gscv.best_params_, logger = logger)
        custom_print('\n---------------------------------\n',logger = logger)
        churn_eval_metrics(xgb_pred, self.y_test, logger)
        custom_print('\n---------------------------------\n',logger = logger)
        custom_print('Threshold parameter tuning\n', logger = logger)
        threshold, accuracy = plot_pr_curve(xgb_proba, self.y_test, plot_path)
        xgb_pred_best = []
        for i in xgb_proba:
            if i>=threshold:
                xgb_pred_best.append(1)
            else:
                xgb_pred_best.append(0)
        custom_print('Prediction using best threshold for accuracy\n-------------------------\n',
                             logger = logger)
        churn_eval_metrics(xgb_pred_best, self.y_test, logger)
        custom_print('Best threshold for accuracy: ' + str(threshold), logger = logger)
        custom_print('Accuracy score at best threshold: ' + str(accuracy), logger = logger)
        if save_model:
            pickle.dump(xgb, open(model_save_loc, 'wb'))
            custom_print('XGBoost model succesfully saved', logger = logger)
        else:
            custom_print('Warning: XGBoost model has NOT been saved', logger = logger)
    
    def knn(self):
        '''
        When 'KNN' is chosen for parameter model_type in the train function,
        this function will be called to train a KNN model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are n_neighbors and weights
        '''
        #Loading training parameter range
        n_range = config_file[model_number]['n_neighbors']
        n = np.arange(n_range[0], n_range[1], n_range[2])
        weight_range = config_file[model_number]['weights']
        weight = weight_range
        knn_grid = {'n_neighbors':n, 'weights':weight}
        #Execute grid search and fit model
        knn = KNeighborsClassifier()
        knn_gscv = GridSearchCV(knn, knn_grid, cv = 2 ,return_train_score = True, verbose=10)
        knn_gscv.fit(self.x_train, self.y_train)
        knn_para = knn_gscv.best_params_
        knn = KNeighborsClassifier(n_neighbors = knn_para.get('n_neighbors'),
                                    weights = knn_para.get('weights'))
        knn.fit(self.x_train, self.y_train)
        knn_pred = knn.predict(self.x_test)
        knn_proba = knn.predict_proba(self.x_test)[:,1]
        custom_print('KNN model succesfully trained\n', logger = logger)
        custom_print(knn_gscv.best_params_, logger = logger)
        custom_print('\n---------------------------------\n',logger = logger)
        churn_eval_metrics(knn_pred, self.y_test, logger)
        custom_print('\n---------------------------------\n',logger = logger)
        custom_print('Threshold parameter tuning\n', logger = logger)
        threshold, accuracy = plot_pr_curve(knn_proba, self.y_test, plot_path)
        knn_pred_best = []
        for i in knn_proba:
            if i>=threshold:
                knn_pred_best.append(1)
            else:
                knn_pred_best.append(0)
        custom_print('Prediction using best threshold for accuracy\n-------------------------\n',
                                logger = logger)
        churn_eval_metrics(knn_pred_best, self.y_test, logger)
        custom_print('Best threshold for accuracy: ' + str(threshold), logger = logger)
        custom_print('Accuracy score at best threshold: ' + str(accuracy), logger = logger)
        if save_model:
            pickle.dump(knn, open(model_save_loc, 'wb'))
            custom_print('K-nearest neighbour model succesfully saved', logger = logger)
        else:
            custom_print('Warning: K-nearest neighbour model has NOT been saved', logger = logger)  


if __name__ == "__main__":
    curr_dir = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_number", type=str, required=True)
    args = parser.parse_args()

    config_path = os.path.join(curr_dir, 'model_config.yml')
    config_file = parse_config(config_path)
    fishing_df = load_dataset(config_file)
    DATE_COLUMNS = config_file["date_columns"]
    TARGET_COLUMN = config_file["target"]
    fishing_df_preprocess = preprocess_data(fishing_df)
    X_train, X_test, Y_train, Y_test = train_test_processing(fishing_df_preprocess,DATE_COLUMNS,TARGET_COLUMN)

    model_number = args.model_number
    model_block = config_file[model_number]
    model_name = model_block['model_name']
    
    model_save_loc = os.path.join(curr_dir, config_file[model_number]['model_save_loc'])
    data_path = config_file['data_path']
    threshold = config_file[model_number]['threshold']
    is_train = config_file[model_number]['is_train']
    save_model = config_file[model_number]['save_model']
    log_path = os.path.join(curr_dir,config_file[model_number]['log_path'] + model_name + ".log")
    plot_path =  os.path.join(curr_dir,config_file[model_number]['plot_path'] + "/" + model_name)
    print(log_path)
    print(plot_path)
    logger = open(os.path.join(curr_dir, log_path), 'w')
    custom_print(f"Model selected.. {model_name}  \n",logger = logger)
    df = FishingModel(X_train, X_test, Y_train, Y_test , model_name = model_name)
    if is_train:
        custom_print("Training dataset has been loaded successfully\n",logger = logger)
        custom_print('---------------------------------',logger = logger)
        custom_print("Start training...\n",logger = logger)
        start_time = time.time()
        df.train(model_name)
        custom_print('\n---------------------------------\n',logger = logger)
    else:
        custom_print("Data to be predicted has been loaded successfully",logger = logger)
        df.predict(model_name, threshold)
    logger.close()