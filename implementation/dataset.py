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
    
class RandomCutBalancedTreeDataset(Dataset):
    def __init__(self, propagation_path, idx_path, embedding_path, num_classes, num_each_class, cut_prob):
        self.num_each_class = num_each_class
        self.num_classes = num_classes
        self.cut_prob = cut_prob
        
        with open(propagation_path, 'rb') as file:
            self.propagations = pickle.load(file)
        with open(idx_path, 'rb') as file:
            self.tid2idx = pickle.load(file)
        self.embeddings = []

        with open(embedding_path, 'r') as file:
            pbar = tqdm(total=len(self.tid2idx))
            pbar.set_description("Loading embeddings")
            while True:
                line = file.readline()
                if not line:
                    break
                embed = np.array(eval(line))
                if embed.shape[0] == 0:
                    embed = np.zeros((1, 300))
                self.embeddings.append(embed)
                pbar.update()

        self.tups = []
        for tweets, edges, label in self.propagations:    
            tup = self.build_tree(tweets, edges)
            self.tups.append(tup)
        
        self.series = []
        for tup in self.tups:
            self.series.append(self.time_series(tup))

        self.sampling_idxs = []
        for target in range(self.num_classes):
            tmp = [idx for idx, (tids, edges, label) in enumerate(self.propagations) if label==target]
            self.sampling_idxs += random.choices(tmp, k = self.num_each_class)
        random.shuffle(self.sampling_idxs)
    
    
    def __len__(self):
        return len(self.sampling_idxs)
    
    def __getitem__(self, index):
        index = self.sampling_idxs[index]
        tweets, edges, label = self.propagations[index]
        tids = [node['tid'] for node in self.propagations[index][0]]
        tup = self.tups[index]
        root, leaf, children, c_count, parents, p_count, tid2labels, tid2time = tup
        # random cut (remove invalid nodes)
        max_t = 7 * 24 * 60
        sampled_t = random.choices(self.cut_prob[0], weights=self.cut_prob[1], k=1)[0]
        valid_nodes = {tweet['tid'] for tweet in tweets if tid2time[tweet['tid']] < sampled_t}
        tweets = [tweet for tweet in tweets if tweet['tid'] in valid_nodes]
        tids = {tweet['tid'] for tweet in tweets}
        edges = [edge for edge in edges if edge[0] in tids and edge[1] in tids]
        tup = self.build_tree(tweets, edges)
        
        tid2idx = {}
        embeddings = []
        for tid in tids:
            tid2idx[tid] = len(embeddings)
            embeddings.append(self.embeddings[self.tid2idx[tid]])
        
        timeseries = self.time_series(tup, sampled_t)
        return tup, tid2idx, embeddings, timeseries, label
    
    def build_tree(self, tweets, edges):
        nodes = set()
        nodes = {tweet['tid'] for tweet in tweets}
        p_count = {node: 0 for node in nodes}
        parents = {node: [] for node in nodes}
        c_count = {node: 0 for node in nodes}
        children = {node: [] for node in nodes}

        for edge in edges:
            child, parent = edge
            p_count[child] += 1
            parents[child].append(parent)
            c_count[parent] += 1
            children[parent].append(child)
        
        leaf = [tid for tid in c_count if c_count[tid]==0]
        root = [tid for tid in p_count if p_count[tid]==0]
        
        tid2labels = {tweet['tid']: tweet['label'] for tweet in tweets}
        
        start_time = min([tweet['created_at'] for tweet in tweets if tweet['tid'] in root])        
        tid2time = {tweet['tid']: (tweet['created_at']- start_time).seconds/60 for tweet in tweets}        
        
        return root, leaf, children, c_count, parents, p_count, tid2labels, tid2time
    
    
    def time_series(self, tup, max_t = 60.):
        # window
        total_time = max_t
        window = 5.
        
        root, leaf, children, c_count, parents, p_count, tid2labels, tid2time = tup
        timeseries = np.zeros([int(total_time//window), 1])
        
        for tid in children:
            timeseries[np.clip(np.round(tid2time[tid]//window).astype(int), 0, timeseries.shape[0]-1)][0] += 1
        return timeseries
    
class FixedCutTreeDataset(Dataset):
    def __init__(self, propagation_path, idx_path, embedding_path, num_classes, cut_length):
        self.num_classes = num_classes
        self.cut_length = cut_length
        
        with open(propagation_path, 'rb') as file:
            self.propagations = pickle.load(file)
        with open(idx_path, 'rb') as file:
            self.tid2idx = pickle.load(file)
        self.embeddings = []

        with open(embedding_path, 'r') as file:
            pbar = tqdm(total=len(self.tid2idx))
            pbar.set_description("Loading embeddings")
            while True:
                line = file.readline()
                if not line:
                    break
                embed = np.array(eval(line))
                if embed.shape[0] == 0:
                    embed = np.zeros((1, 300))
                self.embeddings.append(embed)
                pbar.update()

        self.tups = []
        for tweets, edges, label in self.propagations:    
            tup = self.build_tree(tweets, edges)
            self.tups.append(tup)
        
        self.series = []
        for tup in self.tups:
            self.series.append(self.time_series(tup))
    
    
    def __len__(self):
        return len(self.propagations)
    
    def __getitem__(self, index):
        tweets, edges, label = self.propagations[index]
        tids = [node['tid'] for node in self.propagations[index][0]]
        tup = self.tups[index]
        root, leaf, children, c_count, parents, p_count, tid2labels, tid2time = tup
               
        max_t = self.cut_length
        valid_nodes = {tweet['tid'] for tweet in tweets if tid2time[tweet['tid']] < max_t}
        tweets = [tweet for tweet in tweets if tweet['tid'] in valid_nodes]
        tids = {tweet['tid'] for tweet in tweets}
        edges = [edge for edge in edges if edge[0] in tids and edge[1] in tids]
        tup = self.build_tree(tweets, edges)
        
        tid2idx = {}
        embeddings = []
        for tid in tids:
            tid2idx[tid] = len(embeddings)
            embeddings.append(self.embeddings[self.tid2idx[tid]])
        
        timeseries = self.time_series(tup, self.cut_length)
        return tup, tid2idx, embeddings, timeseries, label
    
    def build_tree(self, tweets, edges):
        nodes = set()
        nodes = {tweet['tid'] for tweet in tweets}
        p_count = {node: 0 for node in nodes}
        parents = {node: [] for node in nodes}
        c_count = {node: 0 for node in nodes}
        children = {node: [] for node in nodes}

        for edge in edges:
            child, parent = edge
            p_count[child] += 1
            parents[child].append(parent)
            c_count[parent] += 1
            children[parent].append(child)
        
        leaf = [tid for tid in c_count if c_count[tid]==0]
        root = [tid for tid in p_count if p_count[tid]==0]
        
        tid2labels = {tweet['tid']: tweet['label'] for tweet in tweets}
        
        start_time = min([tweet['created_at'] for tweet in tweets if tweet['tid'] in root])        
        tid2time = {tweet['tid']: (tweet['created_at']- start_time).seconds/60 for tweet in tweets}        
        
        return root, leaf, children, c_count, parents, p_count, tid2labels, tid2time
    
    def time_series(self, tup, max_t = 60.):
        # window
        total_time = max_t
        window = 5.
        
        root, leaf, children, c_count, parents, p_count, tid2labels, tid2time = tup
        timeseries = np.zeros([int(total_time//window), 1])
        
        for tid in children:
            timeseries[np.clip(np.round(tid2time[tid]//window).astype(int), 0, timeseries.shape[0]-1)][0] += 1
        return timeseries
    
class BalancedFixedCutTreeDataset(Dataset):
    def __init__(self, propagation_path, idx_path, embedding_path, num_classes, num_each_class, cut_length):
        self.num_classes = num_classes
        self.num_each_class = num_each_class
        self.cut_length = cut_length
        
        with open(propagation_path, 'rb') as file:
            self.propagations = pickle.load(file)
        with open(idx_path, 'rb') as file:
            self.tid2idx = pickle.load(file)
        self.embeddings = []

        with open(embedding_path, 'r') as file:
            pbar = tqdm(total=len(self.tid2idx))
            pbar.set_description("Loading embeddings")
            while True:
                line = file.readline()
                if not line:
                    break
                embed = np.array(eval(line))
                if embed.shape[0] == 0:
                    embed = np.zeros((1, 300))
                self.embeddings.append(embed)
                pbar.update()

        self.tups = []
        for tweets, edges, label in self.propagations:    
            tup = self.build_tree(tweets, edges)
            self.tups.append(tup)
        
        self.series = []
        for tup in self.tups:
            self.series.append(self.time_series(tup))
    
        self.sampling_idxs = []
        for target in range(self.num_classes):
            tmp = [idx for idx, (tids, edges, label) in enumerate(self.propagations) if label==target]
            self.sampling_idxs += random.choices(tmp, k = self.num_each_class)
        random.shuffle(self.sampling_idxs)
    
    def __len__(self):
        return len(self.sampling_idxs)
    
    def __getitem__(self, index):
        index = self.sampling_idxs[index]
        tweets, edges, label = self.propagations[index]
        tids = [node['tid'] for node in self.propagations[index][0]]
        tup = self.tups[index]
        root, leaf, children, c_count, parents, p_count, tid2labels, tid2time = tup
               
        max_t = self.cut_length
        valid_nodes = {tweet['tid'] for tweet in tweets if tid2time[tweet['tid']] < max_t}
        tweets = [tweet for tweet in tweets if tweet['tid'] in valid_nodes]
        tids = {tweet['tid'] for tweet in tweets}
        edges = [edge for edge in edges if edge[0] in tids and edge[1] in tids]
        tup = self.build_tree(tweets, edges)
        
        tid2idx = {}
        embeddings = []
        for tid in tids:
            tid2idx[tid] = len(embeddings)
            embeddings.append(self.embeddings[self.tid2idx[tid]])
        
        timeseries = self.time_series(tup, self.cut_length)
        return tup, tid2idx, embeddings, timeseries, label
    
    def build_tree(self, tweets, edges):
        nodes = set()
        nodes = {tweet['tid'] for tweet in tweets}
        p_count = {node: 0 for node in nodes}
        parents = {node: [] for node in nodes}
        c_count = {node: 0 for node in nodes}
        children = {node: [] for node in nodes}

        for edge in edges:
            child, parent = edge
            p_count[child] += 1
            parents[child].append(parent)
            c_count[parent] += 1
            children[parent].append(child)
        
        leaf = [tid for tid in c_count if c_count[tid]==0]
        root = [tid for tid in p_count if p_count[tid]==0]
        
        tid2labels = {tweet['tid']: tweet['label'] for tweet in tweets}
        
        start_time = min([tweet['created_at'] for tweet in tweets if tweet['tid'] in root])        
        tid2time = {tweet['tid']: (tweet['created_at']- start_time).seconds/60 for tweet in tweets}        
        
        return root, leaf, children, c_count, parents, p_count, tid2labels, tid2time
    
    def time_series(self, tup, max_t = 60.):
        # window
        total_time = max_t
        window = 5.
        
        root, leaf, children, c_count, parents, p_count, tid2labels, tid2time = tup
        timeseries = np.zeros([int(total_time//window), 1])
        
        for tid in children:
            timeseries[np.clip(np.round(tid2time[tid]//window).astype(int), 0, timeseries.shape[0]-1)][0] += 1
        return timeseries