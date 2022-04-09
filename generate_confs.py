import numpy as np
import os
import pandas as pd

# base_dir是最原始的pdb文件
base_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/pdb_truncated/'
receptor_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/blast_pdbqt/'
ligand_dir = '/home/kmk_gmx/Desktop/bioinfo/ligand_dock/'
config_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/docking_confs/'
output_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/docking_output/'

source2blast = pd.read_csv('./source2blast.csv')
blastDict = {}
for i, item in source2blast.iterrows():
    blastDict[item['blast']] = item['source']

def getLigandDir(pdb_id):
    return ligand_dir + pdb_id + '.pdbqt'

for pdb_name in os.listdir(receptor_dir):
    pdb_id = pdb_name[3:-6]
    print("pdb_id:", pdb_id)
    pocket_file_name = receptor_dir+pdb_name
    config_file = []
    xs = [1e5,-1e5];ys = [1e5,-1e5];zs = [1e5,-1e5]

    sourceLigandId = blastDict[pdb_id]
    print("source ligand name:", sourceLigandId)
    config_file.append('receptor = '+receptor_dir+pdb_id+'.pdbqt\n')
    config_file.append('ligand = '+ getLigandDir(sourceLigandId) + '\n')

    with open(pocket_file_name) as pocket_file:
        pocket = pocket_file.readlines()
        flag = False
        for line in pocket:
            if line[:4]=='ATOM':
                # print(info)
                x = float(line[30:38]);y=float(line[38:46]);z=float(line[46:54])
                xs[0] = x if xs[0]>x else xs[0];xs[1] = x if xs[0]<x else xs[0]
                ys[0] = y if ys[0]>y else ys[0];ys[1] = y if ys[0]<y else ys[0]
                zs[0] = z if zs[0]>z else zs[0];zs[1] = z if zs[0]<z else zs[0]
        xsize = abs(xs[1]-xs[0]);ysize = abs(ys[1]-ys[0]);zsize = abs(zs[1]-zs[0])
        x_cent = (xs[1]+xs[0])/2;y_cent = (ys[1]+ys[0])/2;z_cent = (zs[1]+zs[0])/2
        config_file.append('center_x = '+str(x_cent)+'\n')
        config_file.append('center_y = '+str(y_cent)+'\n')
        config_file.append('center_z = '+str(z_cent)+'\n')
        config_file.append('size_x = '+str(xsize)+'\n')
        config_file.append('size_y = '+str(ysize)+'\n')
        config_file.append('size_z = '+str(zsize)+'\n')
    config_file.append('energy_range = 4\n')
    config_file.append('exhaustiveness = 9\n')
    config_file.append('num_modes = 9\n')
    config_file.append('out = '+output_dir + pdb_id +'.pdbqt\n')
    with open(config_dir+pdb_id+'.txt', 'w+') as config:
        config.writelines(config_file)
