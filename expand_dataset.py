import pandas as pd
import os

def df2dict(df, key, val):
    res = {}
    for i, row in df.iterrows():
        res[row[key]] = row[val]
    return res

train_pred = df2dict(pd.read_csv('./train_pred.csv'), 'name', 'pred')
valid_pred = df2dict(pd.read_csv('./validation_pred.csv'), 'name', 'pred')
test_pred = df2dict(pd.read_csv('./test_pred.csv'), 'name', 'pred')

nega_dir = '../negative_graph_featureSimplified/'

file_names = []
labels = []
for i in os.listdir(nega_dir):
    if i in train_pred or i in valid_pred:
        continue
    file_names.append(nega_dir + i)
    labels.append(1.0)
	
df = pd.DataFrame({'file_name': file_names, 'label': labels})
df.to_csv('test_simple_expanded')
