# -*- coding: utf-8 -*-
"""
godd

split dataset into train dataset and validation dataset

input: positive_dataset.csv, negative_dataset.csv
output: 
"""

import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle


positive_path = "../positive_graph_featureSimplified/"
negative_path = "../negative_graph_featureSimplified/"
posiFiles = [positive_path + i for i in os.listdir(positive_path)]
negaFiles = [negative_path + i for i in os.listdir(negative_path)]

rmsd_df = pd.read_csv('../dock_rsmd.csv')
dock_rmsd = {}
for i, row in rmsd_df.iterrows():
    dock_rmsd[row['dock_name']] = float(row['rsmd'])
posiLabels = [1.0] * len(posiFiles)
negaLabels = []
for i in negaFiles:
    if dock_rmsd[i[-13:]] < 2.0:
        negaLabels.append(1.0)
    else:
        negaLabels.append(0.0)

dfAll = pd.DataFrame({'file_name':posiFiles + negaFiles, 'label': posiLabels + negaLabels})

dfTrain = pd.DataFrame(columns=['file_name', 'label'])
dfValidation = pd.DataFrame(columns=['file_name', 'label'])
dfTest = pd.DataFrame(columns=['file_name', 'label'])
numTrain = 0
numValidation = 0
numTest = 0
dictTrain = []
dictVali = []
dictTest = []

count = 0
for i, row in dfAll.iterrows():
    print('\r' + str(count), end='')
    count+=1
    name = ''
    #extract name
    if row['label'] == 0.0:
        name = row['file_name'][-13:-9]
    else:
        name = row['file_name'][-4:]
    # add a new row
    if name in dictTrain:
        dfTrain.loc[numTrain] = row
        numTrain += 1
    elif name in dictVali:
        dfValidation.loc[numValidation] = row
        numValidation += 1
    elif name in dictTest:
        dfTest.loc[numTest] = row
        numTest += 1
    else:
        flag = np.array([2*numTrain, 7*numValidation, 14*numTest]).argsort()[0]
        if flag == 0:
            dfTrain.loc[numTrain] = row
            numTrain += 1
            dictTrain.append(name)
        elif flag == 1:
            dfValidation.loc[numValidation] = row
            numValidation += 1
            dictVali.append(name)
        else:
            dfTest.loc[numTest] = row
            numTest += 1
            dictTest.append(name)


print(numTrain, numValidation, numTest)

dfTrain.to_csv('./new_train_dataset_simple.csv')
dfValidation.to_csv('./_new_validation_dataset_simple.csv')
dfTest.to_csv('./_new_test_simple.csv')