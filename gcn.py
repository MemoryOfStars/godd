import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv


class GCN(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCN, self).__init__()
        self.gat = GATConv(in_feats, in_feats, num_heads=1, allow_zero_in_degree=True)
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
        edata = np.array(g.edata['h'].cpu())
        for i, e in enumerate(edata):
            if e != 1.0:
                edata[i] = math.exp(((e-self.param_mu.item())**2)/(-self.param_sigma.item()))
        #pow_param = torch.mul(g.edata['h'] - self.param_mu, g.edata['h'] - self.param_mu)/(-self.param_sigma)
        #efeat = torch.log(pow_param)
        g.edata['h'] = torch.Tensor(edata).cuda()
        # Use GAT
        #h = self.gat(g, inputs)
        #print("shapes----", inputs.shape, h.shape)
        #h = self.conv1(g, h)
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