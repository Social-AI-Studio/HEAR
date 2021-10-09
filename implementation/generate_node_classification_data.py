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

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", dest='k', type=int,help="fold num")
    parser.add_argument("-d", "--dataset", dest='dataset', help="dataset path")
    parser.add_argument("-m", dest='magnitude_file', help="word embedding model file (Magnitude format)")

    args = parser.parse_args()
    dataset_path = args.dataset
    k = args.k
    sample_num = 15000

    
    print('Loading data')
    with open(dataset_path + 'nodes.json', 'r') as file:
        tweets = json.load(file)
    vectors = Magnitude(args.magnitude_file)
    folded_path = dataset_path + 'folded/'
    for i in range(k):
        with open(folded_path + 'propagation_fold%d.pkl'%i, 'rb') as file:
            test_propagations = pickle.load(file)
        test_tweets = {tweet['tid'] for prop in test_propagations for tweet in prop[0]}
        train_tweets = {tid: tweets[tid] for tid in tweets if tid not in test_tweets}
        label2tids = {0: [], 1: []}
        for tid in tqdm(train_tweets):
            label2tids[train_tweets[tid]['label']].append(tid)
        tids = random.sample(label2tids[0], k=sample_num) + random.sample(label2tids[1], k=sample_num)
        data = [(ft_embed(vectors, train_tweets[tid]['text']), train_tweets[tid]['label']) for tid in tqdm(tids)]
        with open(folded_path + 'data_for_node_classification_fold%d.pkl'%i, 'wb') as file:
            pickle.dump(data, file)

if __name__ == '__main__':
    main()