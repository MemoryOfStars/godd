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

device = torch.device("cuda:0")
modelPath = '../models/gcnplus02dataset1654729163.pkl'
model = torch.load(modelPath)

print(model)

#file_path = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/aligned_blast_dgl/'
csv_file_path = './new_test_simple.csv'
df = pd.read_csv(csv_file_path)
file_names = []
predictions = []

#for i, f in enumerate(os.listdir(file_path)):
for i, row in df.iterrows():
    #g = dgl.load_graphs(file_path+f)[0][0].to(device) # load_graphs returns tuple(graphs, labels)
    g = dgl.load_graphs(row['file_name'])[0][0].to(device) # load_graphs returns tuple(graphs, labels)
    pred = model(g, g.ndata['h'].float()).squeeze(1).squeeze(1)
    file_names.append(row['file_name'].split('/')[-1])
    predictions.append(pred.item())
    print('\r' + str(i), pred, end='')


df = pd.DataFrame({"name":file_names, "pred":predictions})
df.to_csv('test_pred_newtrain.csv')
