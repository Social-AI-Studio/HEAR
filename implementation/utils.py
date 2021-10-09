import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

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

from dataset import RandomCutBalancedTreeDataset, FixedCutTreeDataset
from utils import *
from model import *

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
    return sum(acc)/len(acc), sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1), true_accuracy

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
    
def node_collate_fn(data):
    seq, label = zip(*data)
    seq = list(seq)
    label = list(label)
    #print(seq, label)
    pairs = [(s, l) for s, l in zip(seq, label)]
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    seq = [s for s, l in pairs]
    label = [l for s, l in pairs]
    seq_length = [len(sq) for sq in seq]
    seq = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    labels = torch.zeros(0, 1)
    for l in label:
        labels = torch.cat([labels, l.unsqueeze(0)], axis=0)
    return seq, seq_length, labels

def validation_early_detection(model, num_classes, test_loader, cut_length):
    test_loader.dataset.cut_length = cut_length
    model.eval()
    pred_pairs = []
    for idx, (tup, tid2idx, feats, timeseries, label) in enumerate(test_loader):
        root, leaf, children, c_count, parents, p_count, tid2labels, tid2time = tup
        output, node_output, node_labels = model.tree_and_node(tid2idx, feats, leaf, parents, children, timeseries, tid2labels, tid2time)
        pred_pairs.append((target2label(output.tolist()), label.item()))
        del output
    acc, prec, rec, f1, true_acc = metrices(pred_pairs, num_classes)
    print('Validation of %d minutes, Raw Acc: %3f, Balanced Acc: %3f, F1: %3f'%(int(cut_length),true_acc, rec, f1))
    return true_acc, rec, f1


def early_detection_cross_fold(train_datasets, test_datasets, num_folds, test_fold):
    train = [train_datasets[i] for i in range(num_folds) if i!=test_fold]
    train_dataset = ConcatDataset(train)
    test_dataset =  test_datasets[test_fold]
    return train_dataset, test_dataset

def load_train_datasets(train_path, num_folds, num_classes, num_each_class, cut_prob):
    train_datasets = [RandomCutBalancedTreeDataset(train_path + 'propagation_fold%d'%i + '.pkl',
                                                   train_path + 'tid2embed_idx_fold%d'%i + '.pkl',
                                                   train_path+ 'ft_embedding_fold%d'%i + '.txt',
                                                   num_classes,
                                                   num_each_class,
                                                   cut_prob)  for i in range(num_folds)]
    return train_datasets

def load_test_datasets(test_path, num_folds, num_classes, cut_length):
    test_datasets = [FixedCutTreeDataset(test_path + 'propagation_fold%d'%i + '.pkl',
                                          test_path + 'tid2embed_idx_fold%d'%i + '.pkl',
                                          test_path+ 'ft_embedding_fold%d'%i + '.txt',
                                          num_classes,
                                          cut_length)  for i in range(num_folds)]
    return test_datasets