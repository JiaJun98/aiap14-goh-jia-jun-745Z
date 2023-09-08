import os
import sqlite3
import pandas as pd
from utility import parse_config
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np

def check():
    print("Testings123")

def load_dataset(config_file):
    """Establishing connection with sqlite3
    
    Parameters
    ----------
        config_file_path : str
            Config file to read dataset path
     Return
    ------
        fishing_df: DataFrame containing original data
    """
    DATASET_PATH = config_file["data_path"]
    conn = sqlite3.connect(DATASET_PATH)
    fishing_df = pd.read_sql_query("SELECT * FROM fishing", conn)
    fishing_df_copy = fishing_df.copy()
    return fishing_df_copy

def preprocess_data(fishing_df):
    """Conduct pre-processing steps based on EDA
    
    Parameters
    ----------
        fishing_df : Pandas.DataFrame
            Raw dataset for problem statement
     Return
    ------
        cleaned_fishing_df: Cleaned and Transformed dataframe
    """
    #Splitting Date into Year Month, Day
    DatelID = fishing_df.columns.get_loc('Date')
    fishing_df['Date']= pd.to_datetime(fishing_df['Date'],format='%Y-%m-%d')
    fishing_df['Year'] = fishing_df['Date'].dt.year
    fishing_df['Month'] = fishing_df['Date'].dt.month
    fishing_df['Day'] = fishing_df['Date'].dt.day
    fishing_df.insert(DatelID + 1, "Year", fishing_df.pop("Year"))
    fishing_df.insert(DatelID + 1, "Month", fishing_df.pop("Month"))
    fishing_df.insert(DatelID + 1, "Day", fishing_df.pop("Day"))

    #Missing data
    fishing_df['RainToday'].fillna('Unknown', inplace=True)
    fishing_df_copy_no_missing = fishing_df.dropna()
    #Numerical variables
    #Invalid data
    fishing_df_copy_no_missing["Sunshine"] = fishing_df_copy_no_missing["Sunshine"].apply(lambda x: x *(-1) if x < 0 else x )
    fishing_df_copy_no_missing_valid =  fishing_df_copy_no_missing.query('Evaporation >  0')
    transformed_cols = ['WindGustSpeed', 'WindSpeed9am', 'Evaporation', 'WindSpeed3pm','Sunshine']
    #Skewness
    fishing_df_copy_no_missing_transformed = fishing_df_copy_no_missing_valid[transformed_cols].apply(lambda x: np.log1p(x+1))
    non_transformed_columns = fishing_df_copy_no_missing_valid.drop(columns=transformed_cols, inplace = False)
    fishing_df_copy_no_missing_tranformed_full = pd.concat([non_transformed_columns, fishing_df_copy_no_missing_transformed], axis=1)
    #Checking shape
    print(fishing_df_copy_no_missing_tranformed_full.shape)
    #Categorial Variables
    fishing_df_copy_no_missing_tranformed_full["Pressure9am"] = fishing_df_copy_no_missing_tranformed_full["Pressure9am"].apply(lambda x: x.lower())
    fishing_df_copy_no_missing_tranformed_full["Pressure3pm"] = fishing_df_copy_no_missing_tranformed_full["Pressure3pm"].apply(lambda x: x.lower())
    #One Hot encoding
    categorical_cols = ['Location','WindDir9am', 'WindDir3pm','WindGustDir','Pressure9am', 'Pressure3pm', 'RainToday','ColourOfBoats']
    fishing_df_copy_no_missing_tranformed_full_ohc = pd.get_dummies(fishing_df_copy_no_missing_tranformed_full, columns = categorical_cols, dummy_na=False, dtype=int)
    print(f' Final OHC dataframe shape: {fishing_df_copy_no_missing_tranformed_full_ohc.shape}')
    
    fishing_df_copy_no_missing_tranformed_full_ohc["RainTomorrow"] = fishing_df_copy_no_missing_tranformed_full_ohc["RainTomorrow"].apply(lambda x: 1 if x == "Yes" else 0)
    return fishing_df_copy_no_missing_tranformed_full_ohc

def train_test_processing(cleaned_df, date_columns, target_column): #["Date", "Day", "Month", "Year"], 'RainTomorrow'
    """Conduct Train-test split and SMOTE for class imbalance
    
    Parameters
    ----------
        cleaned_df : Pandas.DataFrame
            Cleaned dataset based on EDA
     Return
    ------
        X_train: Training dataset without label
        X_test:  Testing dataset without label
        Y_train: Training label
        Y_test:  Testing label
    """
    fishing_df_copy_no_missing_tranformed_full_ohc_SMOTE = cleaned_df.drop(columns = date_columns, inplace = False)
    X = fishing_df_copy_no_missing_tranformed_full_ohc_SMOTE.drop(target_column, axis=1, inplace = False)
    y = fishing_df_copy_no_missing_tranformed_full_ohc_SMOTE[target_column]
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X, y)
    print("\nConducting SMOTE....")
    print(f'Shape of X before SMOTE: {X.shape}' )
    print(f'Shape of X after SMOTE: {X_sm.shape}')
    print('Balance of positive and negative classes (%):')
    print(y_sm.value_counts(normalize=True) *100)
    fishing_df_copy_no_missing_tranformed_full_ohc_SMOTE_balance = pd.concat([X_sm,y_sm.to_frame()],axis=1)
    print(fishing_df_copy_no_missing_tranformed_full_ohc_SMOTE_balance.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X_sm, y_sm, test_size=0.20, random_state=42)
    return X_train, X_test, Y_train, Y_test