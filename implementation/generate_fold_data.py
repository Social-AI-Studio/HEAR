import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
import csv

import json
from tqdm.auto import tqdm
from datetime import datetime, date, time, timezone, timedelta
import pickle

import math
import random

import networkx as nx
from networkx.algorithms import tree, dag, distance_measures
from pymagnitude import *
import argparse

def sampling(data):
    # ignore extreme data
    droppings = []
    for i in range(len(data)):
        if len(data[i][0]) > 5000:
            droppings.append(data[i])
    for d in droppings:
        data.remove(d)
    #sampling
    label2pids = {0: [], 1: [], 2: [], 3: [], 4: []}
    for idx, (nodes, edges, label) in tqdm(enumerate(data), total=len(data)):
         label2pids[label].append(idx)
    ratios = {
        0: 0.01,
        1: 0.05,
        2: 1,
        3: 1,
        4: 1
    }

    sampled_pids = {}
    for label in label2pids:
        sampled_pids[label] = random.sample(label2pids[label], math.ceil(len(label2pids[label])*ratios[label]))
        
    return sampled_pids

def chunkIt(seq, num):
    random.shuffle(seq)
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def create_folds(pids, num_folds):
    fold_pids = [[] for i in range(num_folds)]
    del pids[4]
    for label in pids:
        chunked = chunkIt(pids[label], num_folds)
        for i in range(num_folds):
            fold_pids[i] += chunked[i]
    return fold_pids

def tokenization(text):
    # text tokenization, get rid of links
    tokens = text.lower().split()
    exclude = "#!\"$%&'()*+,./:;<=>?[\]^_`{|}~-—"
    tokens = [ch.strip(exclude) for ch in tokens]
    tokens = [token for token in tokens if ('https://t.co/' not in token) and (token!='amp')]
    return tokens

def ft_embed(vectors, text):
    embedding = np.zeros((0, vectors.dim))
    text = tokenization(text)
    for word in text:
        if word in vectors:
            embed = np.expand_dims(vectors.query(word), axis=0)
            #print(embed.shape)
            embedding = np.concatenate((embedding, embed), axis=0)
    return embedding

def save_fold_data(path, vectors, name, fold_data, fold_num):
    fold_tids = set()
    for idx, (nodes, edges, label) in enumerate(fold_data):
        for node in nodes:
            fold_tids.add(node['tid'])
    
    fold_embeddings = []
    fold_idx_dict = {}
    counter = 0
    pbar = tqdm(total=len([1 for prop in fold_data for node in prop[0]]))
    with open(path + name + 'ft_embedding_fold%d'%fold_num + '.txt', 'w') as out_file:
        for prop in fold_data:
            for node in prop[0]:
                embed = ft_embed(vectors, node['text'])
                #embed = ft_embed(model, node['text'])
                out_file.write(str(embed.tolist())+'\n')
                pbar.update()
                fold_idx_dict[node['tid']] = counter
                counter += 1
    
    for (nodes, edges, label) in fold_data:
        for node in nodes:
            del node['text']
    
    with open(path + name + 'propagation_fold%d'%fold_num + '.pkl', 'wb') as file:
        pickle.dump(fold_data, file)
    with open(path + name+ 'tid2embed_idx_fold%d'%fold_num + '.pkl', 'wb') as file:
        pickle.dump(fold_idx_dict, file)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", dest='k', type=int,help="fold num")
    parser.add_argument("-d", "--dataset", dest='dataset', help="dataset path")
    parser.add_argument("-m", dest='magnitude_file', help="word embedding model file (Magnitude format)")

    args = parser.parse_args()
    dataset_path = args.dataset
    k = args.k
    
    print('Loading data')
    with open(dataset_path + 'formatted_data.json', 'r') as file:
        data = json.load(file)
    sampled_pids = sampling(data)
    folded = create_folds(sampled_pids, k)
    folded_data = [[data[pid] for pid in folded[fold]] for fold in range(k)]
    vectors = Magnitude(args.magnitude_file)
    name = 'folded/'
    os.makedirs(dataset_path+name, exist_ok=True)

    for fold in range(k):
        save_fold_data(dataset_path, vectors, name, folded_data[fold], fold)

if __name__ == '__main__':
    main()