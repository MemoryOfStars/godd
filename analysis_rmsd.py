import numpy as np
import os
import pandas as pd
from calculate_rmsd import RMSDCalculator

# base_dir是最原始的pdb文件
ligand_dir = '/home/kmk_gmx/Desktop/bioinfo/ligand_dock/'
dockings_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/split_docking/'

source2blast = pd.read_csv('./source2blast.csv')
blastDict = {}
for i, item in source2blast.iterrows():
    blastDict[item['blast']] = item['source']

def getLigandDir(pdb_id):
    return ligand_dir + pdb_id + '.pdbqt'

cal = RMSDCalculator()
rmsds = []
for dock_name in os.listdir(dockings_dir):
    dock_id = dock_name[:4]
    ligandDir = getLigandDir(blastDict[dock_id])
    rmsd = cal.calculateRMSD(ligandDir, dockings_dir+dock_name)
    if rmsd < 10:
        rmsds.append(rmsd)
print(len(rmsds), rmsds[0])
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title("RMSD Trend")
rmsdContFigs = sns.distplot(rmsds, axlabel='RMSD', color="red", ax=ax)

rmsdContFigs.get_figure().savefig('./graphs/blast_docking_rmsd.png', dpi=400)
