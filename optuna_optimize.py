import optuna
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
from dgl.data import DGLDataset
from dgl.dataloading.pytorch import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from my_dataset import  MyDataset
from gcn import GCN

DEVICE = torch.device("cuda:0")

def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    # 隐藏层层数超参数调整
    n_layers = trial.suggest_int("n_layers", 8, 12)
    layers = []

    n_features = 34
    layers.append(n_features)
    for i in range(n_layers):
    # 隐藏层神经元形式调整  [-1,input,output]
        out_features = trial.suggest_int("n_units_l{}".format(i), 32, 200)
        layers.append(out_features)

     return GCN(layers, [], 1)
def objective(trial):

    # Generate the model.
    # 创建包含优化的模型
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    # 创建可选优化器
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # 创建可调整的学习率
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    trainDataset = MyDataset('./train_dataset_simple.csv', batchSize, (-1, -1))
    validationDataset = MyDataset('./validation_dataset_simple.csv', batchSize, (2, 4))
    
    trainSampler = SubsetRandomSampler(torch.arange(numTrain))
    validationSampler = SubsetRandomSampler(torch.arange(numValidation))

    trainDataloader = GraphDataLoader(trainDataset, sampler=trainSampler, batch_size=batchSize, drop_last=False)
    validationDataloader = GraphDataLoader(validationDataset, sampler=validationSampler, batch_size=batchSize, drop_last=False)

    # TODO adjust following and debug
    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

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
