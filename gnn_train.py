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
batchSize = 20
trainName = 'GCNWithNewTestDataset'

from dgl.data import DGLDataset
from my_dataset import  MyDataset

trainDataset = MyDataset('./train_dataset.csv', batchSize)
validationDataset = MyDataset('./validation_dataset.csv', batchSize)

from dgl.dataloading.pytorch import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

numTrain = len(trainDataset)
numValidation = len(validationDataset)
print("numTrain:" + str(numTrain) + ", numValidation:" + str(numValidation))

trainSampler = SubsetRandomSampler(torch.arange(numTrain))
validationSampler = SubsetRandomSampler(torch.arange(numValidation))

trainDataloader = GraphDataLoader(trainDataset, sampler=trainSampler, batch_size=batchSize, drop_last=False)
validationDataloader = GraphDataLoader(validationDataset, sampler=validationSampler, batch_size=batchSize, drop_last=False)


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
for epoch in range(300):
    
    for batched_graph, labels in tqdm(trainDataloader):
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

    print("epochs:"+str(epoch)+"------------------------loss:"+str(temp))
    num_correct = 0
    num_tests = 0
    
    FP = 0
    FN = 0
    for batched_graph, labels in validationDataloader:
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
    # with open("./losses", 'w+') as f:
    # 	f.write(str(temp))
    # with open("./accs", 'w+') as f:
    # 	f.write(str(num_correct/num_tests))
    if epoch%50 == 0:
        version = str(int(time.time()))
        plt.cla()
        plt.plot(losses)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.savefig('./graphs/gcn_losses' + version + trainName + '.png')
        plt.cla()
        plt.plot(test_acc)
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.savefig('./graphs/gcn_testAcc' + trainName + version + '.png')
        plt.cla()
        plt.plot(FPRate)
        plt.xlabel('epochs')
        plt.ylabel('FP Rate')
        plt.savefig('./graphs/gcn_FPRate' + version + trainName + '.png')
        plt.cla()
        plt.plot(FNRate)
        plt.xlabel('epochs')
        plt.ylabel('FN Rate')
        plt.savefig('./graphs/gcn_FNRate' + version + trainName + '.png')
        torch.save(gnn, '../models/gcn' + version + trainName + '.pkl')


version = str(int(time.time()))
plt.cla()
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.savefig('./graphs/gcn_losses' + trainName + version + '.png')
plt.cla()
plt.plot(test_acc)
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.savefig('./graphs/gcn_testAcc' + trainName + version + '.png')
plt.cla()
plt.plot(FPRate)
plt.xlabel('epochs')
plt.ylabel('FP Rate')
plt.savefig('./graphs/gcn_FPRate' + trainName + version + '.png')
plt.cla()
plt.plot(FNRate)
plt.xlabel('epochs')
plt.ylabel('FN Rate')
plt.savefig('./graphs/gcn_FNRate' + trainName + version + '.png')
torch.save(gnn, '../models/gcn' + trainName + version + '.pkl')

