import os
import pandas as pd
import numpy as np
import math

ligand_dir = '/home/kmk_gmx/Desktop/bioinfo/ligand_dock/'
aligned_dock_dir = '/home/kmk_gmx/Desktop/bioinfo/dockings/'

'''
source2blast = pd.read_csv('./source2blast.csv')
blastDict = {}
for i in os.listdir(aligned_dock_dir):
    blastDict[i[:-6]] = i[5:9]
'''

blast_docks = []
rmsds = []
for fname in os.listdir(aligned_dock_dir):
    if len(fname) != len('5orh_ligand_1.pdbqt'):
        continue
    #blast_id = fname[:-6]
    print(fname)
    #lig_name = blastDict[blast_id]
    lig_name = fname[:4]
    oriX = [];oriY = [];oriZ = []
    with open(ligand_dir + lig_name + '.pdbqt') as natFile:
        for line in natFile.readlines():
            if(line[:4] == 'ATOM'):
                oriX.append(float(line[31:38].strip()))
                oriY.append(float(line[39:46].strip()))
                oriZ.append(float(line[47:54].strip()))
    n = len(oriX)
    oriX = np.array(oriX);oriY = np.array(oriY);oriZ = np.array(oriZ)
    
    dockX = [];dockY = [];dockZ = []
    with open(aligned_dock_dir + fname,'r') as dockFile:
        for line in dockFile.readlines():
            if(line[:4] == 'ATOM'):
                dockX.append(float(line[30:38].strip()))
                dockY.append(float(line[38:46].strip()))
                dockZ.append(float(line[46:54].strip()))
    dockX = np.array(dockX);dockY = np.array(dockY);dockZ = np.array(dockZ)
    rmsd = math.sqrt(((dockX-oriX)**2 + (dockY-oriY)**2 + (dockZ-oriZ)**2).sum()/n)
    blast_docks.append(fname[:18])
    rmsds.append(rmsd)

df = pd.DataFrame({'blast':blast_docks, 'rmsd':rmsds})
df.to_csv('./train_rmsds.csv')
