import os
import numpy as np
import math
import pandas as pd

origin_path = '/home/kmk_gmx/Desktop/bioinfo/ligand_dock/'
dock_path = '/data/bilab/kaku/docking/'


docks = os.listdir(dock_path)
csv_names = []
csv_rsmds = []


class RMSDCalculator():
    def __init__(self, naturalFile, dockingFile):
        self.naturalFile = naturalFile
        self.dockingFile = dockingFile
    def calculate(self, output):
        print(self.naturalFile)
        ori_x = [];ori_y = [];ori_z = []
        with open(self.naturalFile) as ori_file:
            for line in ori_file.readlines():
                if(line[:4] == 'ATOM'):
                    ori_x.append(float(line[31:38].strip()))
                    ori_y.append(float(line[39:46].strip()))
                    ori_z.append(float(line[47:54].strip()))
        n = len(ori_x)
        ori_x = np.array(ori_x);ori_y = np.array(ori_y);ori_z = np.array(ori_z)

        num_mode = 1
        while num_mode <= 9 :
            dock_name = origin_name[:4]+'_ligand_'+str(num_mode)+'.pdbqt'
            num_mode += 1
            if dock_name not in docks:
                break
            dock_x = [];dock_y = [];dock_z = []
            with open(dock_path+dock_name,'r') as dock_file:
                for line in dock_file.readlines():
                    if(line[:4] == 'ATOM'):
                        dock_x.append(float(line[31:38].strip()))
                        dock_y.append(float(line[39:46].strip()))
                        dock_z.append(float(line[47:54].strip()))
            dock_x = np.array(dock_x);dock_y = np.array(dock_y);dock_z = np.array(dock_z)

            rmsd = math.sqrt(((dock_x-ori_x)**2 + (dock_y-ori_y)**2 + (dock_z-ori_z)**2).sum()/n)
            csv_names.append(dock_name[:-6])
            csv_rsmds.append(rmsd)

        dataframe = pd.DataFrame({'dock_name':csv_names, 'rsmd':csv_rsmds})
        dataframe.to_csv("dock_rsmd.csv")
