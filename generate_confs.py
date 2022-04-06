import numpy as np
import os

def getName():
	

base_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/pdb_truncated/'
receptor_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/pdb_truncated/'
ligand_dir = '../ligand_dock/'
config_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/pdb_truncated/'
for pdb_name in os.listdir(base_dir):
    print(pdb_name)
    pocket_file_name = base_dir+pdb_name+'/'+pdb_name+'_ligand.mol2'
    config_file = []
    xs = [1e5,-1e5];ys = [1e5,-1e5];zs = [1e5,-1e5]
    config_file.append('receptor = '+receptor_dir+pdb_dir+'.pdbqt\n')
    config_file.append('ligand = '+ligand_dir+pdb_dir+'.pdbqt\n')
    with open(pocket_file_name) as pocket_file:
        pocket = pocket_file.readlines()
        flag = False
        for line in pocket:
            if line[:9]=='@<TRIPOS>' and line[9:13]=='ATOM' and flag==False:
                flag = True
            elif line[:9]=='@<TRIPOS>' and line[9:13]!='ATOM':
                flag = False
            elif flag:
                info = line.split()
                x = float(info[2]);y=float(info[3]);z=float(info[4])
                xs[0] = x if xs[0]>x else xs[0];xs[1] = x if xs[0]<x else xs[0]
                ys[0] = y if ys[0]>y else ys[0];ys[1] = y if ys[0]<y else ys[0]
                zs[0] = z if zs[0]>z else zs[0];zs[1] = z if zs[0]<z else zs[0]
        xsize = 20;ysize = 20;zsize = 20
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
    config_file.append('out = /data/'+pdb_dir+'.pdbqt\n')
    #print(xs,ys,zs)
    with open(config_dir+pdb_dir+'.txt', 'w+') as config:
        config.writelines(config_file)
