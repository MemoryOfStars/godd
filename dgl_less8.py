positivePDBDir = "../positive_truncated_data/"
positiveTarDir = "/data/bilab/kaku/positive_graph_save8A/"
negativePDBDir = "../negative_pdb/"
negativeTarDir = "/data/bilab/kaku/negative_graph_save8A/"

valid_elements = ['N', 'C', 'O', 'S', 'H', 'P', 'F', 'CL',  'BR',  'ZN']
element_onehot = {}
for idx, ele in enumerate(valid_elements):
    element_onehot[ele] = [0]*len(valid_elements)
    element_onehot[ele][idx] = 1
def ligandCenter(pdbData):
    minX, minY, minZ = 99999.0, 99999.0, 99999.0
    maxX, maxY, maxZ = -99999.0, -99999.0, -99999.0
    for data in pdbData:
        if data[1] == 0:     # Ignore Protein Atoms
            continue
        minX = data[3] if data[3]<minX else minX
        minY = data[4] if data[4]<minY else minY
        minZ = data[5] if data[5]<minZ else minZ
        maxX = data[3] if data[3]>maxX else maxX
        maxY = data[4] if data[4]>maxY else maxY
        maxZ = data[5] if data[5]>maxZ else maxZ
    return minX, minY, minZ, maxX, maxY, maxZ

'''
def fartherThan8A(ligandBox, atomPosition):
    x, y, z = atomPosition[0], atomPosition[1], atomPosition[2]
    minX, minY, minZ, maxX, maxY, maxZ = ligandBox
    if minX<x<maxX and minY<y<maxY and minZ<z<maxZ:
        return False
    x_dis = x - minX
    y_dis = y - minY
    z_dis = z - minZ
    disMin = math.sqrt(x_dis**2+y_dis**2+z_dis**2)

    x_dis = x - maxX
    y_dis = y - maxY
    z_dis = z - maxZ
    disMax = math.sqrt(x_dis**2+y_dis**2+z_dis**2)
    return disMin>8 and disMax>8
'''
    
def fartherThan8A(ligandBox, atomPosition):
    x, y, z = atomPosition[0], atomPosition[1], atomPosition[2]
    minX, minY, minZ, maxX, maxY, maxZ = ligandBox

    centerX, centerY, centerZ = (minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2
    disMin = math.sqrt((centerX-minX)**2+(centerY-minY)**2+(centerZ-minZ)**2)
    disMax = math.sqrt((centerX-maxX)**2+(centerY-maxY)**2+(centerZ-maxZ)**2)
    radius = disMin if disMin > disMax else disMax
    
    dis = math.sqrt((centerX-x)**2+(centerY-y)**2+(centerZ-z)**2)
    return (dis-radius)>8

from dgl.data import DGLDataset
import os
import numpy as np
import pandas as pd
import torch as th
import dgl
import json
import math

for pdb_name in os.listdir(positivePDBDir):
    pdb_data = []                        #一个pdb文件中的信息汇总
    # [element_name, protein_or_ligand, chain_id, x, y, z]
    with open(positivePDBDir+pdb_name) as pdb_file:
        lines = pdb_file.readlines()
        flag = False
        for i, line in enumerate(lines):
            if (line[:4]=='ATOM' or line[:6]=='HETATM') and line[76:78].strip() in valid_elements:
                atom = ['C', 0, 'A', 0.0, 0.0, 0.0]#0代表蛋白质分子。 1代表ligand分子
                atom[0] = line[76:78].strip()                   #element_name
                atom[1] = 0 if line[:4]=='ATOM' else 1  #protein or ligand
                if (i > 0 and int(line[6:11]) < int(lines[i-1][6:11])) or flag:
                    flag = True
                    atom[1] = 1
                atom[2] = line[21]                      #chain_id
                atom[3] = float(line[30:38].strip())    #x
                atom[4] = float(line[38:46].strip())    #y
                atom[5] = float(line[46:54].strip())    #z
                pdb_data.append(atom)
    ligandBox = ligandCenter(pdb_data)
    graph_edata = [] #edge weights
    u_set = []
    v_set = []
    graph_ndata = []
    nodeDict = {}
    pos = 0
    for i in range(len(pdb_data)):
        if fartherThan8A(ligandBox, [pdb_data[i][3], pdb_data[i][4], pdb_data[i][5]]):
            continue
        graph_ndata.append([element_onehot[pdb_data[i][0]]]) #[[ele1, other_feature], [ele2, other_feature],....]
        nodeDict[i] = pos
        pos += 1
    for i in nodeDict.items():
        for j in nodeDict.items():
            x_dis = pdb_data[i[0]][3] - pdb_data[j[0]][3]
            y_dis = pdb_data[i[0]][4] - pdb_data[j[0]][4]
            z_dis = pdb_data[i[0]][5] - pdb_data[j[0]][5]
            dis = math.sqrt(x_dis**2+y_dis**2+z_dis**2)
            if dis < 2.0 and pdb_data[i[0]][1]==pdb_data[j[0]][1] and pdb_data[i[0]][2] == pdb_data[j[0]][2]:
                graph_edata.append([1])
                graph_edata.append([1])
                u_set.append(i[1])
                v_set.append(j[1])
                u_set.append(j[1])
                v_set.append(i[1])
            elif dis < 5.0 and (pdb_data[i[0]][1]==pdb_data[j[0]][1] or pdb_data[i[0]][2] == pdb_data[j[0]][2]):
                graph_edata.append([dis])
                graph_edata.append([dis])
                u_set.append(i[1])
                v_set.append(j[1])
                u_set.append(j[1])
                v_set.append(i[1])
    g = dgl.DGLGraph()
    g.add_nodes(len(graph_ndata))
    g.add_edges(u_set, v_set)
    print(pdb_name, len(pdb_data), len(graph_edata), len(graph_ndata), len(u_set), len(v_set))
    g.edata['h'] = th.tensor(graph_edata)
    g.ndata['h'] = th.tensor(graph_ndata)

    dgl.save_graphs(positiveTarDir+pdb_name[:-4], [g])
