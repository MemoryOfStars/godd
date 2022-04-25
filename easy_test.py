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
modelPath = '../models/gcn1649678138newLayers.pkl'
model = torch.load(modelPath)

print(model)

#file_path = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_dgl/'
file_path = '/home/kmk_gmx/Desktop/bioinfo/positive_graph_featureSimplified/'
file_names = []
predictions = []

for i, f in enumerate(os.listdir(file_path)):
    g = dgl.load_graphs(file_path+f)[0][0].to(device) # load_graphs returns tuple(graphs, labels)
    pred = model(g, g.ndata['h'].float()).squeeze(1).squeeze(1)
    file_names.append(f)
    predictions.append(pred.round().item())
    #print('\r' + str(i), pred, end='')

print(predictions.count(1.0)/len(predictions))

#df = pd.DataFrame({"name":file_names, "pred":predictions})
#df.to_csv('blast_pred.csv')
