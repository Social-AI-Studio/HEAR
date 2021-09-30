import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import numpy as np
from utils import *

class MLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.linear2 = nn.Linear(mid_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, GRU_layers):
        super(SimpleRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.GRU_layers = GRU_layers
        self.proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.GRU = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.GRU_layers, bias=True, batch_first=True, bidirectional=True)
        self.output = nn.Linear(self.hidden_dim*2, self.out_dim) # in_dim is hidden_dim*2 since we have bidirectional GRU
        self.relu = nn.ReLU()
    
    def get_embedding(self, seq, seq_length):        
        x = self.proj(seq)
        x = self.relu(x)
        x = rnn_utils.pack_padded_sequence(x, seq_length, batch_first=True, enforce_sorted=False)
        output, h_n = self.GRU(x)
        # output: [batch_size , seq_len, hidden_dim], h_n: [num_layers, batch_size, hidden_dim]
        out_pad, out_len = rnn_utils.pad_packed_sequence(output, batch_first=True)
        x = out_pad[torch.arange(out_len.shape[0]), out_len-1, :]
        x = self.relu(x)
        return x
    
class NodeEmbed(nn.Module):
    def __init__(self, batch_size, in_dim, hidden_dim):
        super(NodeEmbed, self).__init__()
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.RNN = SimpleRNN(self.in_dim, self.hidden_dim, 2, 1)
        self.fc = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
    def forward(self, tid2idx, embeddings):
        max_len = max(max([embedding.shape[1] for embedding in embeddings]), 1)
        embed_mtrx = torch.zeros(0, max_len, self.in_dim).cuda()
        seq_length = []
        for embedding in embeddings:
            if embedding.shape[1] == 0:
                embedding = torch.zeros(1, 1, self.in_dim)
            embed_mtrx = torch.cat([embed_mtrx, F.pad(embedding.float().cuda(), (0, 0, 0, max_len - embedding.shape[1]), "constant", 0)])
            seq_length.append(embedding.shape[1])
        
        idx = 0
        tmp_embed = torch.zeros(0, self.hidden_dim).cuda()
        while idx < len(tid2idx):
            # set up start and end idx
            start_idx = idx
            idx += self.batch_size
            idx = min(idx, len(tid2idx))
            rnn_embed = self.RNN.get_embedding(embed_mtrx[start_idx: idx, :, :], seq_length[start_idx: idx])
            rnn_embed = self.fc(rnn_embed)
            tmp_embed = torch.cat([tmp_embed, rnn_embed], axis=0)
        
        tid2embeddings = {tid: tmp_embed[tid2idx[tid]].unsqueeze(0) for tid in tid2idx}
        del embed_mtrx,tmp_embed
        return tid2embeddings

class CNN_RNN(nn.Module):
    def __init__(self, hidden):
        super(CNN_RNN, self).__init__()
        self.conv = nn.Conv1d(2, hidden, kernel_size=3)
        self.rnn = nn.GRU(hidden, hidden*2, 1, bidirectional=False, batch_first=False)
        self.linear = nn.Linear(hidden*2, hidden*4)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        _, x = self.rnn(x)
        x = self.linear(x.squeeze(0))
        return x

class Attention(nn.Module):
    def __init__(self, in_dim, dimensions):
        super(Attention, self).__init__()

        self.in_dim = in_dim
        self.dimensions = dimensions
        self.k_proj = nn.Linear(in_dim, dimensions, bias=False)
        self.q_proj = nn.Linear(in_dim, dimensions, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Sigmoid()

    def forward(self, query, context):
        batch_size, q_len, in_dim = query.size()
        output_len = context.size(1)
        
        query = query.reshape(batch_size * q_len, in_dim)
        query = self.q_proj(query)
        query = query.reshape(batch_size, q_len, self.dimensions)

        context = context.reshape(batch_size * output_len, in_dim)
        context = self.k_proj(context)
        context = context.reshape(batch_size, output_len, self.dimensions)
        
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        attention_weights = self.tanh(attention_scores)
        return attention_weights

class Up_RvNN(nn.Module):
    def __init__(self, node_dim , hidden_dim, attn_dim):
        super(Up_RvNN, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.GRU = torch.nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bias=True)
        self.proj = nn.Linear(self.node_dim, self.hidden_dim)
        self.attention = Attention(self.hidden_dim, self.attn_dim)
        
    def traverse(self, tid2embeddings, source, asc_dict, desc_dict):
        nodes_h = {tid: torch.zeros(0).cuda() for tid in tid2embeddings}
        current_nodes = [node.item() for node in source]
        root_h = torch.zeros(0, self.hidden_dim).cuda()
        # start from leaf nodes
        # we need to make sure all the child of a node had been 
        while True:
            expansions = []
            for node in current_nodes:
                #### forward computing
                if len(desc_dict[node]) == 0:
                    # leaf nodes, zeros as previous hidden
                    childs_h = torch.zeros(1, self.hidden_dim).float().cuda()
                else:
                    # normal nodes, previous hidden is sum of hidden of its children
                    tmp_h = torch.zeros(0, self.hidden_dim).cuda()
                    for child in desc_dict[node]:
                        tmp_h = torch.cat([tmp_h, nodes_h[child.item()]], 0)
                    childs_h = tmp_h

                # forward GRU with previous hidden and node's embedding
                p_embed = self.proj(tid2embeddings[node].squeeze(0).float().cuda())
                weights = self.attention(childs_h.unsqueeze(0), p_embed.unsqueeze(0))
                weights = weights.view(1, -1)
                output = torch.mm(weights, childs_h)
                nodes_h[node] = self.GRU(p_embed, output)
                # store root nodes
                if len(asc_dict[node]) == 0:
                    root_h = torch.cat([root_h, nodes_h[node]], 0)
            # check which nodes to expand
            for node in nodes_h:
                flag = True
                # if node had been visited, ignore it
                if nodes_h[node].shape[0] != 0:
                    continue

                # chekc all the children had been visited
                for child in desc_dict[node]:
                    if nodes_h[child.item()].shape[0] == 0:
                        flag = False
                        break

                # expand unvisited nodes of which children had all been visited
                if flag:
                    expansions.append(node)
            current_nodes = expansions
            # exapansion loop terminated
            if len(current_nodes) == 0:
                break
        del nodes_h
        return root_h
    
    def forward(self, tid2embeddings, source, asc_dict, desc_dict):
        root_h = self.traverse(tid2embeddings, source, asc_dict, desc_dict)
        output, _ =  torch.max(root_h, 0)
        del root_h
        return output
    


class NodeClassy(nn.Module):
    def __init__(self, feat_dim, batch_size):
        super(NodeClassy, self).__init__()
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        
        self.classifier = MLP(self.feat_dim, self.feat_dim//8, 2)
    
    def forward(self, tid2embeddings, tid2label):
        embeddings = torch.zeros(0, self.feat_dim).cuda()
        labels = torch.zeros(0).long().cuda()
        tids = []
        
        for tid in tid2embeddings:
            embeddings = torch.cat([embeddings, tid2embeddings[tid].squeeze(0)], axis=0)
            labels = torch.cat([labels, tid2label[tid].cuda()])
            tids.append(tid)
        
        idx = 0
        output = torch.zeros(0, 2).cuda()
        while idx < embeddings.shape[0]:
            # set up start and end idx
            start_idx = idx
            idx += self.batch_size
            idx = min(idx, embeddings.shape[0])
            tmp_output = self.classifier(embeddings[start_idx: idx, :])
            output = torch.cat([output, tmp_output], axis=0)
        
        return output, labels, tids

class Network(nn.Module):
    def __init__(self, feat_dim, tree_hidden, time_hidden, node_batch_size, num_classes):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.tree_hidden = tree_hidden
        self.time_hidden = time_hidden
        
        self.node_batch_size = node_batch_size
        
        self.node_embed = NodeEmbed(batch_size = self.node_batch_size, in_dim = self.feat_dim, hidden_dim = self.tree_hidden)
        
        self.rvnn = Up_RvNN(self.tree_hidden, self.tree_hidden, 256)
        
        self.time_net = CNN_RNN(self.time_hidden)
        
        self.classifier = MLP(self.tree_hidden + 4*self.time_hidden, (self.tree_hidden + 4*self.time_hidden)//4, self.num_classes)        
        self.node_classifier = NodeClassy(self.tree_hidden, self.node_batch_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def tree_and_node(self, tid2idx, feats, leaf, parents, children, timeseries, tid2labels, tid2time):
        tid2embeddings = self.node_embed(tid2idx, feats)
        tree_h = self.rvnn(tid2embeddings, leaf, parents, children)
        
        node_output, node_labels, tids = self.node_classifier(tid2embeddings, tid2labels)
        timeseries = self.add_time_channel(timeseries, tids, tid2time, node_output)
        time_h = self.time_net(timeseries.cuda().float())
        output = torch.cat([tree_h, time_h.squeeze()])
        pred = self.classifier(output)

        return pred, node_output, node_labels
    
    def add_time_channel(self, timeseries, tids, tid2time, node_output):
        # window
        window = 5.
        #with torch.no_grad():
        second_channel = torch.zeros_like(timeseries)
        for idx in range(len(tids)):
            slot = (tid2time[tids[idx]]//window).long().item()
            slot = min(max(0, slot), timeseries.shape[1]-1)
            second_channel[0][slot][0] += self.softmax(node_output[idx])[1].item()
        
        timeseries = torch.cat([timeseries, second_channel], axis=-1)
        
        return timeseries