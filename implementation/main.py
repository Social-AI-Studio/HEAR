import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import numpy as np

import os
import gc

import json
from tqdm.auto import tqdm
import pickle
import argparse
import copy

import math

from dataset import RandomCutBalancedTreeDataset, FixedCutTreeDataset, NodeClassiDataset
from utils import *
from model import *

def warm_up(fold, dataset_path):
    # warm up for node classification module
    with open(dataset_path + 'folded/data_for_node_classification_fold%d.pkl'%fold, 'rb') as file:
        node_classification_data = pickle.load(file)
    nc_dataset = NodeClassiDataset(node_classification_data)
    nc_loader = DataLoader(nc_dataset, batch_size=16, shuffle=True, collate_fn=node_collate_fn, drop_last=False)
    nc_net = SimpleRNN(input_dim=300, hidden_dim=1024, out_dim=2, GRU_layers=1).cuda()
    optimizer = torch.optim.SGD(nc_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=None).cuda()
    for epoch in tqdm(range(20)):
        for idx, (seq, seq_length, labels) in enumerate(nc_loader):
            optimizer.zero_grad()
            seq = seq.cuda()
            labels = labels.cuda()
            out = nc_net(seq, seq_length)
            loss = criterion(out, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()
    return copy.deepcopy(nc_net.state_dict())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", dest='k', type=int,help="fold num")
    parser.add_argument("-d", "--dataset", dest='dataset', help="dataset path")
    parser.add_argument("-e", "--epoch", dest='epochs', type=int, help="traianing epoch num")

    args = parser.parse_args()

    num_classes = 4
    cut_prob = ([60*1., 60*5., 60*12., 60*24., 60*48., 60*72.],
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    val_time = [15., 30., 60., 180., 360., 720., 1440.]

    train_datasets = load_train_datasets(train_path = args.dataset + 'folded/',
                                num_folds = args.k,
                                num_classes = num_classes,
                                num_each_class = 200,
                                cut_prob = cut_prob)

    test_datasets = load_test_datasets(test_path = args.dataset + 'folded/',
                                       num_folds = args.k,
                                       num_classes = num_classes,
                                       cut_length = 60.)

    for fold in range(args.k):
        test_fold = fold
        train_dataset, test_dataset = early_detection_cross_fold(train_datasets, test_datasets, args.k, test_fold)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        model = Network(feat_dim = 300,
                        tree_hidden = 1024,
                        time_hidden = 32,
                        node_batch_size = 32,
                        num_classes = num_classes).cuda()
        
        model.node_embed.RNN.load_state_dict(warm_up(fold, args.dataset))

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
        for epoch in range(args.epochs):
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
            
            for val_t in val_time:
                raw_acc, bal_acc, f1 = validation_early_detection(model, num_classes, test_loader, cut_length = val_t)

if __name__ == '__main__':
    main()