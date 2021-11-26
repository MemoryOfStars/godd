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

from sklearn.utils import shuffle
my_batch_size = 20

from dgl.data import DGLDataset
from my_dataset import  MyDataset

my_dataset = MyDataset('./positive_dataset.csv', './negative_dataset.csv')

from dgl.dataloading.pytorch import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(my_dataset)
print("dataset length:", num_examples)
num_train = int(num_examples*0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(my_dataset, sampler=train_sampler, batch_size=my_batch_size, drop_last=False)
test_dataloader = GraphDataLoader(my_dataset, sampler=test_sampler, batch_size=my_batch_size, drop_last=False)


from gcn import GCN

device = torch.device("cuda:0")
gnn = GCN(44, 1).to(device)


import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

optimizer = torch.optim.Adadelta(gnn.parameters(), lr=0.01)
all_logits = []
losses = []
test_acc = []
FPs = []
FNs = []
FPRate = []
FNRate = []
temp = 0.0
lossFunc = torch.nn.BCELoss()
for epoch in range(800):
    
    for batched_graph, labels in tqdm(train_dataloader):
        # try:
        batched_graph, labels = batched_graph.to(device), labels.to(device)
        pred = gnn(batched_graph, batched_graph.ndata['h'].float()).squeeze(1).squeeze(1)
        # print(pred, labels)
        loss = lossFunc(pred, labels)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp = loss
        # except RuntimeError as exception:
        #     if "out of memory" in str(exception):
        #         print("WARN: out of memory")
        #         gc.collect()
        #         torch.cuda.empty_cache()
        #     else:
        #         continue
    print("epochs:"+str(epoch)+"------------------------loss:"+str(temp))
    num_correct = 0
    num_tests = 0
    #with torch.no_grad():
    
    FP = 0
    FN = 0
    for batched_graph, labels in test_dataloader:
        batched_graph, labels = batched_graph.to(device), labels.to(device)
        pred = gnn(batched_graph, batched_graph.ndata['h'].float()).squeeze(1).squeeze(1)
        # print(pred, labels)
        for i, p in enumerate(pred.round()):
            if p != labels[i]:
                FP += 1 if p == torch.tensor(1.0) else 0
                FN += 1 if p == torch.tensor(0.0) else 0

        num_correct += (pred.round() == labels).sum().item() # TP+TN
        num_tests += len(labels) # TP+TN+FP+FN
    losses.append(temp)
    FPs.append(FP)
    FNs.append(FN)
    curAcc = num_correct/num_tests
    FPRate.append(FP/num_tests)
    FNRate.append(FN/num_tests)
    test_acc.append(curAcc)
    print("epochs:"+str(epoch)+"------------------------Acc:"+str(curAcc) + " FNRate:" + str(FN/num_tests))
    with open("./losses", 'w+') as f:
    	f.write(str(temp))
    with open("./accs", 'w+') as f:
    	f.write(str(num_correct/num_tests))


version = str(int(time.time()))
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.savefig('./graphs/gcn_losses' + version + '.png')
plt.cla()
plt.plot(test_acc)
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.savefig('./graphs/gcn_testAcc' + version + '.png')
plt.cla()
plt.plot(FPRate)
plt.xlabel('epochs')
plt.ylabel('FP Rate')
plt.savefig('./graphs/gcn_FPRate' + version + '.png')
plt.cla()
plt.plot(FNRate)
plt.xlabel('epochs')
plt.ylabel('FN Rate')
plt.savefig('./graphs/gcn_FNRate' + version + '.png')
torch.save(gnn, '../models/gcn' + version + '.pkl')

