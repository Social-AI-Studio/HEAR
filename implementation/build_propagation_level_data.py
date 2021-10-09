import pandas as pd
import numpy as np
import os

import json
from tqdm.auto import tqdm
from datetime import datetime, date, time, timezone, timedelta

import json
import argparse

import networkx as nx
from networkx.algorithms import tree, dag, distance_measures

def load_data(dataset_path):
    with open(dataset_path + 'nodes.json') as handle:
        nodes = json.load(handle)
    with open(dataset_path + 'edges.json') as handle:
        edges = json.load(handle)
    # filter out alone nodes
    contained = set()
    for edge in edges:
        contained.add(edge[0])
        contained.add(edge[1])
    nodes = {int(node_id): nodes[node_id] for node_id in tqdm(nodes) if int(node_id) in contained}
    return nodes, edges

def build_graph(nodes, edges):
    G = nx.DiGraph()
    for start, end in tqdm(edges):
        if (start in nodes) and (end in nodes):
            G.add_edge(start, end, weight=1)
    return G

# utility functions
def find_source(comp, G):
    tmpG = G.subgraph(comp).reverse()
    return list(nx.topological_sort(tmpG))[0]

def before_t(comp, G, tweets, t):
    start_time = tweets[find_source(comp, G)]['timestamp']
    end_time = start_time + t
    return [tid for tid in comp if tweets[tid]['timestamp'] < end_time]

def get_hate_count(comp, tweets):
    return len([1 for tid in comp if tweets[tid]['label']])

def get_level_perc(num_hate, count2percentile):
    if num_hate == 0:
        return 0
    if count2percentile[num_hate] < 0.9:
        return 1
    if count2percentile[num_hate] < 0.99:
        return 2
    return 3

def propagation_level_pair(comp, G, t_k, tweets, count2percentile):
    cut_off_comp = before_t(comp, G, tweets, t_k)
    level = get_level_perc(get_hate_count(cut_off_comp, tweets), count2percentile)

    included_tweets = []
    for tid in cut_off_comp:
        tmp = tweets[tid]
        tmp['tid'] = tid
        included_tweets.append(tmp)
    
    edges = G.subgraph(cut_off_comp).edges
    edges = [(edge[0], edge[1]) for edge in edges]

    return (included_tweets, edges, level)

def get_count2percentile(comps, G, tweets, t_k):
    counts = [get_hate_count(before_t(comp, G, tweets, t_k), tweets) for comp in comps]
    srt_counts = sorted(counts)
    srt_counts = [count for count in srt_counts if count!=0]
    count2percentile = {}
    cur = 0
    for idx, i in enumerate(srt_counts):
        if i > cur:
            count2percentile[i] = idx/len(srt_counts)
            cur = i
    for i in range(min(count2percentile.keys()), max(count2percentile.keys())):
        if i not in count2percentile:
            count2percentile[i] = count2percentile[i-1]
    return count2percentile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", dest='k', type=int,help="label target evaluate days")
    parser.add_argument("-d", "--dataset", dest='dataset', help="dataset path")
    args = parser.parse_args()
    dataset_path = args.dataset
    k = args.k
    t_k = k*86400
    
    print('Loading data')
    nodes, edges = load_data(dataset_path)
    print('Computing labels')
    G = build_graph(nodes, edges)
    comps = list(nx.connected_components(G.to_undirected()))
    tid2cid = {tid: i for i in range(len(comps)) for tid in comps[i]}
    count2percentile = get_count2percentile(comps, G, nodes, t_k)
    print('Building propagation data')
    formatted_data = [propagation_level_pair(comp, G, t_k, nodes, count2percentile) for comp in tqdm(comps)]
    print('Savingn data to %s'%dataset_path)
    with open(dataset_path + 'formatted_data.json', 'w') as handle:
        json.dump(formatted_data, handle)

if __name__ == '__main__':
    main()