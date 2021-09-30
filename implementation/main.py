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

from sklearn.metrics import plot_confusion_matrix
import seaborn as sn

from dataset import RandomCutBalancedTreeDataset, FixedCutTreeDataset
from utils import *
from model import *

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

def main():
    num_classes = 4
    cut_prob = ([60*1., 60*5., 60*12., 60*24., 60*48., 60*72.],
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])


    train_datasets = load_train_datasets(train_path = '../data/',
                                num_folds = 5,
                                num_classes = num_classes,
                                num_each_class = 10,
                                cut_prob = cut_prob)

    test_datasets = load_test_datasets(test_path = '../data/',
                                       num_folds = 5,
                                       num_classes = num_classes,
                                       cut_length = 60.)

    raw_accs =  [[] for i in range(7)]
    bal_accs =  [[] for i in range(7)]
    f1s =  [[] for i in range(7)]

    num_classes = 4
    num_folds = 5
    epochs = 60
    for fold in range(1):
        test_fold = fold
        train_dataset, test_dataset = early_detection_cross_fold(train_datasets, test_datasets, num_folds, test_fold)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        model = Network(feat_dim = 300,
                        tree_hidden = 1024,
                        time_hidden = 32,
                        node_batch_size = 32,
                        num_classes = num_classes).cuda()

        LR = 0.0003
        optimizer = torch.optim.SGD([{'params': model.rvnn.parameters(), 'lr': LR},
                                     {'params': model.node_embed.parameters(), 'lr': LR*0.1},
                                     {'params': model.classifier.parameters(), 'lr': LR},
                                     {'params': model.node_classifier.parameters(), 'lr': LR},
                                     {'params': model.time_net.parameters(), 'lr': LR}],
                                    momentum=0.9, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1. for i in range(num_classes)]).cuda())
        node_criterion = nn.CrossEntropyLoss().cuda()

        losses = []
        for epoch in range(epochs):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            pbar.set_description('Epoch %d'%(epoch+1))
            model.train()
            for idx, (tup, tid2idx, feats, timeseries, label) in pbar:
                optimizer.zero_grad()

                root, leaf, children, c_count, parents, p_count, tid2labels, tid2time = tup
                pred, node_output, node_labels = model.tree_and_node(tid2idx, feats, leaf, parents, children, timeseries, tid2labels, tid2time)

                loss = criterion(pred.unsqueeze(0), label.cuda()) + node_criterion(node_output, node_labels)

                losses.append(loss.item())
                if len(losses) > len(train_loader):
                    losses.pop(0)
                pbar.set_postfix({'Loss': sum(losses)/len(losses)})
                loss.backward()
                optimizer.step()


            raw_acc, bal_acc, f1 = validation_early_detection(model, num_classes, test_loader, cut_length = 60.*1)

            print('Epoch%d, f1: %f, acc: %f'%(epoch, f1, bal_acc))

if __name__ == '__main__':
    main()
