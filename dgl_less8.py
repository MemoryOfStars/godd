from dgl.data import DGLDataset
import os
import numpy as np
import pandas as pd
import torch as th
import dgl
import json
import math
from calculate_rmsd import RMSDCalculator

receptorPDBQTDir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_truncated_pdbqt/'
ligandPDBQTDir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_pdbqt_dock/'      # ligand Dir (file name eg:5orh.pdbqt)
outputGraphDir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/aligned_blast_dgl/' 

receptorFileNames = os.listdir(receptorPDBQTDir)
ligandFileNames = os.listdir(ligandPDBQTDir)
ligPrefixDict = {}
for ligFname in ligandFileNames:
    if ligFname[:9] in ligPrefixDict:
        ligPrefixDict[ligFname[:9]].append(ligandPDBQTDir + ligFname)
        continue
    ligPrefixDict[ligFname[:9]] = []
def getAllLigNames(recepId):
    if recepId not in ligPrefixDict:
        return []
    return ligPrefixDict[recepId]
    '''
    for ligFname in ligandFileNames:
        if recepId in ligFname:
            results.append(ligandPDBQTDir + ligFname)
    return results
    '''

generateFilePairs = []
cal = RMSDCalculator()
for recep in os.listdir(receptorPDBQTDir):
    recepId = recep[:9]
    recepFilePath = receptorPDBQTDir + recep
    curLigandFilePaths = getAllLigNames(recepId)
    for fname in curLigandFilePaths:
        # rmsd = cal.calculateRMSD(recepFile, ligandFile)
        generateFilePairs.append((recepFilePath, fname))
    
def extractAtomLines(pdbqtLines):
    atomLines = []
    for line in pdbqtLines:
        l = line.strip()
        if l[:4] != 'ATOM':
            continue
        pdbqtCols = []
        pdbqtCols.append(l[0:6].strip())
        pdbqtCols.append(l[6:11].strip())
        pdbqtCols.append(l[12:16].strip())
        pdbqtCols.append(l[16:17].strip())
        pdbqtCols.append(l[17:21].strip())
        pdbqtCols.append(l[21:22].strip())
        pdbqtCols.append(l[22:26].strip())
        pdbqtCols.append(l[26:27].strip())
        pdbqtCols.append(l[30:38].strip())
        pdbqtCols.append(l[38:46].strip())
        pdbqtCols.append(l[46:54].strip())
        pdbqtCols.append(l[54:60].strip())
        pdbqtCols.append(l[60:66].strip())
        pdbqtCols.append(l[66:76].strip())
        pdbqtCols.append(l[76:].strip())
        if pdbqtCols[PDBQT_TYPE_INDEX].upper() not in ELEMENT_TYPE:
            continue
        atomLines.append(pdbqtCols)
    return atomLines

PDBQT_X_POS_INDEX = 8
PDBQT_Y_POS_INDEX = 9
PDBQT_Z_POS_INDEX = 10
PDBQT_TYPE_INDEX  = 14
ELEMENT_TYPE = ['HD', 'C', 'A', 'N', 'NA', 'OA', 'F', 'MG', 'P', 'S', 'CL', 'CA', 'MN', 'FE', 'ZN', 'BR', 'I']
RECEP_ELEMENT_ONEHOT = {}
LIG_ELEMENT_ONEHOT = {}
for idx, ele in enumerate(ELEMENT_TYPE):
    RECEP_ELEMENT_ONEHOT[ele] = [0]*len(ELEMENT_TYPE)*2
    RECEP_ELEMENT_ONEHOT[ele][idx] = 1
for idx, ele in enumerate(ELEMENT_TYPE):
    LIG_ELEMENT_ONEHOT[ele] = [0]*len(ELEMENT_TYPE)*2
    LIG_ELEMENT_ONEHOT[ele][idx+len(ELEMENT_TYPE)] = 1


def distanceIn2Atoms(atom1, atom2):
    atom1x = float(atom1[PDBQT_X_POS_INDEX])
    atom1y = float(atom1[PDBQT_Y_POS_INDEX])
    atom1z = float(atom1[PDBQT_Z_POS_INDEX])
    atom2x = float(atom2[PDBQT_X_POS_INDEX])
    atom2y = float(atom2[PDBQT_Y_POS_INDEX])
    atom2z = float(atom2[PDBQT_Z_POS_INDEX])
    return math.sqrt((atom1x-atom2x)**2 + (atom1y-atom2y)**2 + (atom1z-atom2z)**2)

def distancesFromLigand(atom, ligand):
    atomx = float(atom[PDBQT_X_POS_INDEX])
    atomy = float(atom[PDBQT_Y_POS_INDEX])
    atomz = float(atom[PDBQT_Z_POS_INDEX])
    distances = []
    for a in ligand:
        ax = float(a[PDBQT_X_POS_INDEX])
        ay = float(a[PDBQT_Y_POS_INDEX])
        az = float(a[PDBQT_Z_POS_INDEX])
        dist = math.sqrt((atomx-ax)**2 + (atomy-ay)**2 + (atomz-az)**2)
        distances.append(dist)
    return distances

def minDisFromLigand(atom, ligand):
    distances = distancesFromLigand(atom, ligand)
    return min(distances)

def generateDGL(receptor, ligand, name):
    u_set = []
    v_set = []
    graph_ndata = []
    graph_edata = [] #edge weights
    for atom in receptor:
        if atom[PDBQT_TYPE_INDEX].upper() not in RECEP_ELEMENT_ONEHOT:
            print(atom[PDBQT_TYPE_INDEX])
        #print(atom)
        graph_ndata.append([RECEP_ELEMENT_ONEHOT[atom[PDBQT_TYPE_INDEX].upper()]])
    for atom in ligand:
        if atom[PDBQT_TYPE_INDEX].upper() not in RECEP_ELEMENT_ONEHOT:
            print(atom[PDBQT_TYPE_INDEX])
        graph_ndata.append([LIG_ELEMENT_ONEHOT[atom[PDBQT_TYPE_INDEX].upper()]]) 
        
    for i, atomI in enumerate(receptor + ligand):
        ligI  = False if i < len(receptor) else True  # whether it's a ligand atom
        for j, atomJ in enumerate(receptor + ligand):
            ligJ  = False if j < len(receptor) else True
            dis = distanceIn2Atoms(atomI, atomJ)
            if dis < 2.0 and ligI == ligJ:
                graph_edata.append([1])
                u_set.append(i)
                v_set.append(j)
            elif dis < 5.0 and ligI != ligJ:
                graph_edata.append([dis])
                u_set.append(i)
                v_set.append(j)
    g = dgl.DGLGraph()
    g.add_nodes(len(graph_ndata))
    g.add_edges(u_set, v_set)
    print(name, len(graph_ndata), len(graph_edata), len(graph_ndata), len(u_set), len(v_set))
    g.edata['h'] = th.tensor(graph_edata)
    g.ndata['h'] = th.tensor(graph_ndata)

    dgl.save_graphs(name, [g])

def generatePosiDGL(recep, lig, name):
    recepAtomIn8 = []
    for atom in recep:
        if minDisFromLigand(atom, lig) > 8:
            continue
        recepAtomIn8.append(atom)
    generateDGL(recepAtomIn8, lig, name)

def generateNegaDGL(recep, docks, dockNames):
    if len(docks) == 0:
        return
    for i, dock in enumerate(docks):
        recepAtomIn8 = []
        for atom in recep:
            if minDisFromLigand(atom, dock)>8:
                continue
            recepAtomIn8.append(atom)
        generateDGL(recepAtomIn8, dock, negativeTarDir + dockNames[i])


# generate test dataset
logFile = open('./aligned_dgl.log', 'w+')
for pair in generateFilePairs:
    recepFile = open(pair[0])
    ligFile   = open(pair[1])
    name = pair[1][-19:-6]
    
    print(name, file=logFile)
    recep = extractAtomLines(recepFile.readlines())
    lig   = extractAtomLines(ligFile.readlines())
    
    generatePosiDGL(recep, lig, outputGraphDir + name)
    #generateNegaDGL(recep, docks, dockNames)
    
    # print(name[:4], len(docks))
    ligFile.close()
    recepFile.close()
