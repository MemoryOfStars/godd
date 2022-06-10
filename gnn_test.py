import time
import os
import pandas as pd
import numpy as np
import torch
import gc
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.autograd import Variable
from dgl.data import DGLDataset
from sklearn.utils import shuffle
from gcn import GCN
from my_dataset import MyDataset

my_batch_size = 30
my_dataset = MyDataset('./test_simple.csv', my_batch_size, (2, 4))

from dgl.dataloading.pytorch import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(my_dataset)
print("dataset length:", num_examples)

test_sampler = SubsetRandomSampler(torch.arange(num_examples))
test_dataloader = GraphDataLoader(my_dataset, sampler=test_sampler, batch_size=my_batch_size, drop_last=False)

modelPath = '../models/gcnnewLayers1649822481.pkl'
model = torch.load(modelPath)

print(model)

num_correct = 0
num_tests = 0
FP = 0
FN = 0
device = torch.device("cuda:0")
for batched_graph, labels in test_dataloader:
    batched_graph, labels = batched_graph.to(device), labels.to(device)
    pred = model(batched_graph, batched_graph.ndata['h'].float()).squeeze(1).squeeze(1)
    # print(pred, labels)
    for i, p in enumerate(pred.round()):
        if p != labels[i]:
            FP += 1 if p == torch.tensor(1.0) else 0
            FN += 1 if p == torch.tensor(0.0) else 0

    num_correct += (pred.round() == labels).sum().item() # TP+TN
    num_tests += len(labels) # TP+TN+FP+FN
print(num_correct/num_tests)
