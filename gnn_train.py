import time
import os
import pandas as pd
import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from sklearn.utils import shuffle
my_batch_size = 3

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
            self.graph_labels.append(torch.Tensor([1,0])) #positive
            self.graph_labels.append(torch.Tensor([0,1])) #negative
            
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
        return graph[0], label[0].long()

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
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_size, num_classes, allow_zero_in_degree=True)
        param_mu = torch.tensor(0.0)
        param_sigma = torch.tensor(1.0)
        self.param_mu = nn.Parameter(param_mu)
        self.param_sigma = nn.Parameter(param_sigma)

    def forward(self, g, inputs):
        pow_param = torch.mul(g.edata['h'] - self.param_mu, g.edata['h'] - self.param_mu)/(-self.param_sigma)
        efeat = torch.log(pow_param)
        g.edata['h'] = efeat
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
gnn = GCN(10, 16, 2)


import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
all_logits = []
losses = []
test_acc = []
temp = 0.0
for epoch in range(30):
    for batched_graph, labels in tqdm(train_dataloader):
        pred = gnn(batched_graph, batched_graph.ndata['h'].float())
        #print(pred.shape)
        #print(labels.shape)
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp = loss
        print("epochs:"+str(epoch)+"------------------------loss:"+str(loss))
    num_correct = 0
    num_tests = 0
    for batched_graph, labels in test_dataloader:
        pred = gnn(batched_graph, batched_graph.ndata['h'].float())
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
    losses.append(temp)
    test_acc.append(num_correct/num_tests)
    with open("./losses", 'w') as f:
    	f.write(str(temp))
    with open("./accs", 'w') as f:
    	f.write(str(num_correct/num_tests))


version = str(int(time.time()))
plt.plot(losses)
plt.savefig('./graphs/gcn_losses' + version + '.png')
plt.cla()
plt.plot(test_acc)
plt.savefig('./graphs/gcn_testAcc' + version + '.png')
torch.save(gnn, '../models/gcn' + version + '.pkl')

