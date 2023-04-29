import os
import sqlite3
import pandas as pd
from utility import parse_config
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from imblearn.over_sampling import SMOTE
import statsmodels.api as sma

def load_dataset(config_file_path):
    """Set seed for reproducibility.
    
    Parameters
    ----------
        config_file_path : str
            Config file to read dataset path
     Return
    ------
        fishing_df: DataFrame containing original data
    """
    
    #Establishing connection with sqlite3

    config_file = parse_config(config_file_path)
    DATASET_PATH = config_file["data_path"]
    conn = sqlite3.connect(DATASET_PATH)
    fishing_df = pd.read_sql_query("SELECT * FROM fishing", conn)
    fishing_df_copy = fishing_df.copy()
    return fishing_df_copy

if __name__ == "__main__":
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'model_config.yml')
    fishing_df = load_dataset(config_path)