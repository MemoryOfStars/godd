from dgl.data import DGLDataset
import os
import numpy as np
import pandas as pd
import torch as th
import dgl
import math

positive_pdb_dir = '../negative_pdb/'
target_graph_dir = '../negative_graph_save/'
valid_elements = ['N', 'C', 'O', 'S', 'H', 'P', 'F', 'CL',  'BR',  'ZN']
element_onehot = {}.fromkeys(valid_elements, 0 )
element_onehot['N'] = [1,0,0,0,0,0,0,0,0,0]
element_onehot['C'] = [0,1,0,0,0,0,0,0,0,0]
element_onehot['O'] = [0,0,1,0,0,0,0,0,0,0]
element_onehot['S'] = [0,0,0,1,0,0,0,0,0,0]
element_onehot['H'] = [0,0,0,0,1,0,0,0,0,0]
element_onehot['P'] = [0,0,0,0,0,1,0,0,0,0]
element_onehot['F'] = [0,0,0,0,0,0,1,0,0,0]
element_onehot['CL'] = [0,0,0,0,0,0,0,1,0,0]
element_onehot['BR'] = [0,0,0,0,0,0,0,0,1,0]
element_onehot['ZN'] = [0,0,0,0,0,0,0,0,0,1]
for pdb_name in os.listdir(positive_pdb_dir):
    pdb_data = []                        #一个pdb文件中的信息汇总
    # [element_name, protein_or_ligand, chain_id, x, y, z]
    with open(positive_pdb_dir+pdb_name) as pdb_file:
        for i in pdb_file.readlines():
            if (i[:4]=='ATOM' or i[:6]=='HETATM') and i[76:78].strip() in valid_elements:
                temp = ['C', 0, 'A', 2.2, 3.3, 2.2]#0代表蛋白质分子。 1代表ligand分子
                temp[0] = i[76:78].strip()                   #element_name
                temp[1] = 0 if i[:4]=='ATOM' else 1  #protein or ligand
                temp[2] = i[21]                      #chain_id
                temp[3] = float(i[30:38].strip())    #x
                temp[4] = float(i[38:46].strip())    #y
                temp[5] = float(i[46:54].strip())    #z
                pdb_data.append(temp)
    graph_edata = [] #edge weights
    u_set = []
    v_set = []
    graph_ndata = []
    for i in range(len(pdb_data)):
        graph_ndata.append([element_onehot[pdb_data[i][0]]]) #[[ele1, other_feature], [ele2, other_feature],....]
        for j in range(len(pdb_data)):
            x_dis = pdb_data[i][3] - pdb_data[j][3]
            y_dis = pdb_data[i][3] - pdb_data[j][3]
            z_dis = pdb_data[i][3] - pdb_data[j][3]
            dis = math.sqrt(x_dis**2+y_dis**2+z_dis**2)
            if dis < 2.0 and pdb_data[i][1]==pdb_data[j][1] and pdb_data[i][2] == pdb_data[j][2]:
                graph_edata.append([1])
                graph_edata.append([1])
                u_set.append(i)
                v_set.append(j)
                u_set.append(j)
                v_set.append(i)
            elif dis < 5.0 and (pdb_data[i][1]==pdb_data[j][1] or pdb_data[i][2] == pdb_data[j][2]):
                graph_edata.append([dis])
                graph_edata.append([dis])
                u_set.append(i)
                v_set.append(j)
                u_set.append(j)
                v_set.append(i)
    g = dgl.DGLGraph()
    g.add_nodes(len(graph_ndata))
    g.add_edges(u_set, v_set)
    
    g.edata['h'] = th.tensor(graph_edata)
    g.ndata['h'] = th.tensor(graph_ndata)
    print(pdb_name)
    dgl.save_graphs(target_graph_dir+pdb_name[:-4], [g])
