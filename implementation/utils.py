import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
import gc
import csv

import json
from tqdm.auto import tqdm
from datetime import datetime, date, time, timezone, timedelta
import pickle
import random

import math

import networkx as nx
from networkx.algorithms import tree, dag, distance_measures

from sklearn.metrics import plot_confusion_matrix
import seaborn as sn

def metrices(pred_pairs, num_classes):
    acc = []
    precision = []
    recall = []
    f1 = []
    
    for target in range(num_classes):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for pair in pred_pairs:
            if pair[0]==target and pair[1]==target:
                TP += 1
            elif pair[0]==target and pair[1]!=target:
                FP += 1
            elif pair[0]!=target and pair[1]!=target:
                TN += 1
            elif pair[0]!=target and pair[1]==target:
                FN += 1
        acc.append((TP+TN)/(TP+FP+TN+FN+1e-10))
        precision.append(TP/(TP+FP+1e-10))
        recall.append(TP/(TP+FN+1e-10))
        f1.append(2*(precision[target]*recall[target])/(precision[target]+recall[target]+1e-10))
        
    true_accuracy = sum([1 for pair in pred_pairs if pair[0]==pair[1]])/len(pred_pairs)
    
    macro_prec = sum(precision)/len(precision)
    macro_recall = sum(recall)/len(recall)
    macro_f1 = 2*(macro_prec*macro_recall)/(macro_prec+macro_recall+1e-10)
    
    return sum(acc)/len(acc), macro_prec, macro_recall, macro_f1, true_accuracy

# Utility functions
def argmin(lst):
    return min(range(len(lst)), key=lst.__getitem__)

def argmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)

def label2target(label):
    targets = {
        0: torch.tensor([0., 0., 0., 0.]),
        1: torch.tensor([1., 0., 0., 0.]),
        2: torch.tensor([1., 1., 0., 0.]),
        3: torch.tensor([1., 1., 1., 0.]),
        4: torch.tensor([1., 1., 1., 1.])
    }
    return targets[label]

def target2label(target):
    """
    if round(target[3]) == 1:
        return 4
    elif round(target[2]) == 1:
        return 3
    elif round(target[1]) == 1:
        return 2
    elif round(target[0]) == 1:
        return 1
    else:
        return 0
    """
    return argmax(target)

def checking_rec(pred_pairs):
    for target in range(5):
        print(target, len([1 for pair in pred_pairs if pair[0]==pair[1] and pair[1]==target])/len([1 for pair in pred_pairs if pair[1]==target]), len([1 for pair in pred_pairs if pair[0]==pair[1] and pair[1]==target]), len([1 for pair in pred_pairs if pair[1]==target]))
        
def confusion_matrix(pred_pairs, num_classes):
    conf = np.zeros((num_classes, num_classes))
    for pair in pred_pairs:
        conf[pair[0]][pair[1]] += 1
    norm_vec = np.sum(conf, axis=0)
    conf = np.around(conf/norm_vec, decimals=2)
    #df_cm = pd.DataFrame(conf, [cat2player(i) for i in range(num_classes)], [cat2player(i) for i in range(num_classes)])
    df_cm = pd.DataFrame(conf, [i for i in range(num_classes)], [i for i in range(num_classes)])
    
    fig, ax = plt.subplots(figsize=(7.5,5))  
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap='GnBu', fmt='g', ax=ax)
    b, t = ax.get_ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_title("Confusion Matrix of Level Prediction")
    ax.set_ylim(b, t) # update the ylim(bottom, top) values
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_tick_params(rotation=45)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Prediction')
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    ax.title.set_fontsize(16)
    plt.show()