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

class MyDataset(DGLDataset):
    """
    Parameters
    -------------------------
    raw_dir: str
        Specifying the directory that already stores the input data.
    
    """
    _pos_directory= '../positive_graph_save/'
    _neg_directory= '../negative_graph_save/'
    def __init__(self, 
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(MyDataset, self).__init__(name='docking_classify',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
    def download(self):
        pass

    #must be implemented
    def process(self):
        df_pos = pd.read_csv('./positive_dataset.csv')
        df_neg = pd.read_csv('./negative_dataset.csv')
        pos_graphs = df_pos['file_name']
        pos_labels = df_pos['label']
        neg_graphs = df_neg['file_name']
        neg_labels = df_neg['label']

        #half_batch = int(my_batch_size/2)
        self.graph_dataset = []
        self.graph_labels = []
        #negative graphs are more
        for i in range(len(neg_graphs)):
            self.graph_dataset.append(pos_graphs[i%len(pos_graphs)])
            self.graph_dataset.append(neg_graphs[i])
            self.graph_labels.append(torch.Tensor([1])) #positive
            self.graph_labels.append(torch.Tensor([0])) #negative
            
        self.df_dataset = pd.DataFrame({'file_name':self.graph_dataset, 'label':self.graph_labels})
        self.df_dataset = shuffle(self.df_dataset)
        #for i in range(len())

    
    #must be implemented
    def __getitem__(self, idx):
        """get one item by index
        
        Parameters
        ---------------
        idx: int
            Item index

        Returns
        ---------------
        (dgl.DGLGraph, Tensor)
        """
        graph = dgl.load_graphs(self.df_dataset['file_name'][idx.item()])[0] #idx.item():convert torch.Tensor to int
        #print(self.df_dataset['file_name'][idx.item()])
        label = self.df_dataset['label'][idx.item()]
        return graph[0], label[0].float()

    #must be implemented
    def __len__(self):
        #number of data examples
        return self.df_dataset.shape[0]
        

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass

my_dataset = MyDataset()

from dgl.dataloading.pytorch import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(my_dataset)
print("dataset length:", num_examples)
num_train = int(num_examples*0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(my_dataset, sampler=train_sampler, batch_size=my_batch_size, drop_last=False)
test_dataloader = GraphDataLoader(my_dataset, sampler=test_sampler, batch_size=my_batch_size, drop_last=False)


class GCN(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, 80, allow_zero_in_degree=True)
        self.conv2 = GraphConv(80, 160, allow_zero_in_degree=True)
        self.conv3 = GraphConv(160, 112, allow_zero_in_degree=True)
        self.conv4 = GraphConv(112, 160, allow_zero_in_degree=True)
        self.conv5 = GraphConv(160, 176, allow_zero_in_degree=True)
        self.conv6 = GraphConv(176, 96, allow_zero_in_degree=True)
        self.conv7 = GraphConv(96, 144, allow_zero_in_degree=True)
        self.conv8 = GraphConv(144, 96, allow_zero_in_degree=True)
        self.conv9 = GraphConv(96, 128, allow_zero_in_degree=True)
        self.conv10 = GraphConv(128, 96, allow_zero_in_degree=True)
        self.conv11 = GraphConv(96, 160, allow_zero_in_degree=True)
        self.dnn1 = torch.nn.Linear(160, 140)
        self.dnn2  = torch.nn.Linear(140, num_classes)
        param_mu = torch.tensor(0.0)
        param_sigma = torch.tensor(1.0)
        self.param_mu = nn.Parameter(param_mu)
        self.param_sigma = nn.Parameter(param_sigma)

    def forward(self, g, inputs):
        pow_param = torch.mul(g.edata['h'] - self.param_mu, g.edata['h'] - self.param_mu)/(-self.param_sigma)
        efeat = torch.log(pow_param)
        g.edata['h'] = efeat
        h = self.conv1(g, inputs)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.conv3(g, h)
        h = F.leaky_relu(h)
        h = self.conv4(g, h)
        h = F.leaky_relu(h)
        h = self.conv5(g, h)
        h = F.leaky_relu(h)
        h = self.conv6(g, h)
        h = F.leaky_relu(h)
        h = self.conv7(g, h)
        h = F.leaky_relu(h)
        h = self.conv8(g, h)
        h = F.leaky_relu(h)
        h = self.conv9(g, h)
        h = F.leaky_relu(h)
        h = self.conv10(g, h)
        h = F.leaky_relu(h)
        h = self.conv11(g, h)
        h = F.leaky_relu(h)
        g.ndata['h'] = h
        h = dgl.mean_nodes(g, 'h')
        h = F.leaky_relu(h)
        h = self.dnn1(h)
        h = F.dropout(h, p=0.3)
        h = F.leaky_relu(h)
        h = self.dnn2(h)
        h = F.dropout(h, p=0.2)
        h = torch.sigmoid(h)
        return h

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

