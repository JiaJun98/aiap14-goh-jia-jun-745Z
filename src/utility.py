import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    
    Parameters
    ----------
        seed_value : int
            Seed value to be set. Default value is 42
    """
    random.seed(seed_value)
    np.random.seed(seed_value)

def parse_config(config_file):
    """Parsing function for YAML file

    Parameters
    ----------
        config_file : str
        Path of YAML file

    Return
    ------
        config: Nested Python Dictionary containing various information in YAML file
    """
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

def custom_print(*msg, logger):
    """Prints a message and uses a global variable, logger, to save the message
    
    Parameters
    ------
    msg : 
       Message to be logged
    logger : FileObject
        Logger to create logs
    Return
    ------
        None
    """
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))