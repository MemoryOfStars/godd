import optuna
import time
from tqdm import tqdm
import itertools
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
from dgl.data import DGLDataset
from dgl.dataloading.pytorch import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from my_dataset import  MyDataset
from gcn import GCN

DEVICE = torch.device("cuda:0")
lossFunc = torch.nn.BCELoss()
batchSize = 20

def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    # 隐藏层层数超参数调整
    n_layers = trial.suggest_int("n_layers", 8, 12)
    layers = []

    n_features = 34
    layers.append(n_features)
    for i in range(n_layers):
    # 隐藏层神经元形式调整  [-1,input,output]
        out_features = trial.suggest_int("n_units_l{}".format(i), 16, 200)
        layers.append(out_features)
    out_features = trial.suggest_int("n_units_l{}".format(i), 16, 200)
    layers.append(32)
    return GCN(layers, [], 1)
def objective(trial):

    # Generate the model.
    # 创建包含优化的模型
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    # 创建可选优化器
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "Adadelta"])
    # 创建可调整的学习率
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    trainDataset = MyDataset('./new_train_dataset_simple.csv', batchSize, (-1, -1))
    validationDataset = MyDataset('./new_validation_dataset_simple.csv', batchSize, (0, 4))
    numTrain = 2000
    numValidation = 400
    
    trainSampler = SubsetRandomSampler(torch.arange(numTrain))
    validationSampler = SubsetRandomSampler(torch.arange(numValidation))

    trainDataloader = GraphDataLoader(trainDataset, sampler=trainSampler, batch_size=batchSize, drop_last=False)
    validationDataloader = GraphDataLoader(validationDataset, sampler=validationSampler, batch_size=batchSize, drop_last=False)

    # Training of the model.
    for epoch in range(300):
        for batched_graph, labels in tqdm(trainDataloader):
            # try:
            batched_graph, labels = batched_graph.to(DEVICE), labels.to(DEVICE)
            pred = model(batched_graph, batched_graph.ndata['h'].float()).squeeze(1).squeeze(1)
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
    
        for batched_graph, labels in validationDataloader:
            batched_graph, labels = batched_graph.to(DEVICE), labels.to(DEVICE)
            pred = model(batched_graph, batched_graph.ndata['h'].float()).squeeze(1).squeeze(1)

            num_correct += (pred.round() == labels).sum().item() # TP+TN
            num_tests += len(labels) # TP+TN+FP+FN
        curAcc = num_correct/num_tests
        trial.report(curAcc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return curAcc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
