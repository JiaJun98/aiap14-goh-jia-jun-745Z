import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc, f1_score, confusion_matrix,precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

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

def churn_eval_metrics(Y_pred, Y_test, logger):
    """ Logs the accuracy, precision, AUC, F1-Score, Sensitivity, Specificity between predicted and test
    
    Parameters
    ------
    y_pred : list of int
        Predicted values
    y_test : list of int
            Actual values
    logger : FileObject
        Logger to create logs

    Return
    ------
        None
    """

    model_acc = accuracy_score(Y_test, Y_pred)
    model_prec = precision_score(Y_test, Y_pred)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label = 1)
    model_auc = auc(fpr,tpr)

    model_f1score = f1_score(Y_test, Y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/(tp+fn)

    #Change to 2 dp print("{:.2f}".format(x))
    custom_print("model_accuracy: ", "{:.2f}".format(model_acc), logger = logger)
    custom_print("model_precision: ","{:.2f}".format(model_prec), logger = logger)
    custom_print("model_auc: ", "{:.2f}".format(model_auc), logger = logger)
    custom_print("model_f1score: ", "{:.2f}".format(model_f1score), logger = logger)
    custom_print("sensitivity: ", "{:.2f}".format(sensitivity), logger = logger)
    custom_print("specificity: ", "{:.2f}".format(specificity), logger = logger)

def plot_roc_curve(Y_pred, Y_test,plotting_dir):
    """ PLotting ROC curve for predictions by Machine Learning/Deep Learning Model
    
    Parameters
    ------
    y_pred : list of int
         Predicted values
    Y_test : list of int
            Actual values
    plotting_dir : str
        Directory to plot the ROC curve

    Return
    ------
        None
    """

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label = 1)
    f1_scores = [f1_score(Y_test, Y_pred >= t) for t in thresholds]
    accuracy_scores = [accuracy_score(Y_test, Y_pred >= t) for t in thresholds]
    precision_scores = [precision_score(Y_test, Y_pred >= t) for t in thresholds]

    # Find the threshold values that maximize F1, accuracy, and precision
    best_f1_threshold = thresholds[np.argmax(f1_scores)]
    best_acc_threshold = thresholds[np.argmax(accuracy_scores)]
    best_precision_threshold = thresholds[np.argmax(precision_scores)]

    # Plot the ROC curve
    plt.plot(fpr, tpr)

    # Label the points for the best F1, best accuracy, and best precision
    plt.plot(fpr[np.argmax(f1_scores)], tpr[np.argmax(f1_scores)], 'o',
             label=f'Best F1 ({best_f1_threshold:.2f})')
    plt.plot(fpr[np.argmax(accuracy_scores)], tpr[np.argmax(accuracy_scores)], 'o',
             label=f'Best Accuracy ({best_acc_threshold:.2f})')
    plt.plot(fpr[np.argmax(precision_scores)], tpr[np.argmax(precision_scores)], 'o',
             label=f'Best Precision ({best_precision_threshold:.2f})')

    # Add labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(plotting_dir + '_roc_curve')
    plt.close()

def plot_pr_curve(Y_pred, Y_test, plotting_dir):
    """ PLotting Precision-Recall curve for predictions by Machine Learning/Deep Learning Model
    
    Parameters
    ------
    y_pred : list of int
         Predicted values
    Y_test : list of int
            Actual values
    plotting_dir : str
        Directory to plot the ROC curve

    Return
    ------
        None
    """

    precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)
    accuracy_scores = [accuracy_score(Y_test, Y_pred >= t) for t in thresholds]
    f1_scores = [f1_score(Y_test, Y_pred >= t) for t in thresholds]

    # Find the threshold values that maximize F1, accuracy
    best_acc_threshold = thresholds[np.argmax(accuracy_scores)]
    best_f1_threshold = thresholds[np.argmax(f1_scores)]

    # Plot the Precision-Recall curve
    plt.plot(precision, recall)

    # Label the points for the best F1, accuracy
    plt.plot(precision[np.argmax(accuracy_scores)], recall[np.argmax(accuracy_scores)], 'o',
             label=f'Best Accuracy ({best_acc_threshold:.4f})')
    plt.plot(precision[np.argmax(f1_scores)], recall[np.argmax(f1_scores)], 'o',
             label=f'Best F1 ({best_f1_threshold:.4f})')


    # Add labels and legend
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(plotting_dir + '_pr_curve')
    plt.close()

    # Returns best threshold for f1, accuracy
    return best_acc_threshold, max(accuracy_scores)