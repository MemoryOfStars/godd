import pandas as pd

rmsd_csv = './train_rmsds.csv'
train_csv = './train_dataset_simple.csv'
new_train_csv = './new_train_dataset_simple.csv'

rmsd_dict = {}
rmsd_df = pd.read_csv(rmsd_csv)
for i, row in rmsd_df.iterrows():
    rmsd_dict[row['blast'][:-5]] = float(row['rmsd'])

train_df = pd.read_csv(train_csv)
file_name = []
labels = []
for i, row in train_df.iterrows():
    if 'negative' in row['file_name']:
        dock_id = row['file_name'][-13:]
        print(dock_id)
        label = float(row['label'])
        if rmsd_dict[dock_id] < 2.0:
            label = 1.0
    else:
        label = float(row['label'])
    file_name.append(row['file_name'])
    labels.append(label)

new_df = pd.DataFrame({'file_name':file_name, 'label': labels})
new_df.to_csv(new_train_csv)

