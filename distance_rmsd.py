import os
import numpy as np
import pandas as pd
import json
import math

PDBQT_X_POS_INDEX = 8
PDBQT_Y_POS_INDEX = 9
PDBQT_Z_POS_INDEX = 10
PDBQT_TYPE_INDEX  = 14
ELEMENT_TYPE = ['HD', 'C', 'A', 'N', 'NA', 'OA', 'F', 'MG', 'P', 'S', 'CL', 'CA', 'MN', 'FE', 'ZN', 'BR', 'I']

class DistanceRmsd():
    def __init__(self):
        pass
    def rmsd(self, recep, lig):
        self.recep = open(recep, 'r').readlines()
        self.lig = open(lig, 'r').readlines()
        recep_data = self.extractAtomLines(self.recep)
        lig_data = self.extractAtomLines(self.lig)
        recep_graph_data = []
        for row in recep_data:
            if self.minDisFromLigand(row, lig_data) > 8:
                continue
            recep_graph_data.append(row)

        dis_rmsd = 0
        pair_count = 0
        for rrow in recep_graph_data:
            for lrow in lig_data:
                dis_rmsd += self.distanceIn2Atoms(rrow, lrow)**2
                pair_count += 1

        return math.sqrt(dis_rmsd/pair_count)


    def extractAtomLines(self, pdbqtLines):
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

    def distanceIn2Atoms(self, atom1, atom2):
        atom1x = float(atom1[PDBQT_X_POS_INDEX])
        atom1y = float(atom1[PDBQT_Y_POS_INDEX])
        atom1z = float(atom1[PDBQT_Z_POS_INDEX])
        atom2x = float(atom2[PDBQT_X_POS_INDEX])
        atom2y = float(atom2[PDBQT_Y_POS_INDEX])
        atom2z = float(atom2[PDBQT_Z_POS_INDEX])
        return math.sqrt((atom1x-atom2x)**2 + (atom1y-atom2y)**2 + (atom1z-atom2z)**2)
    
    def distancesFromLigand(self, atom, ligand):
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
    
    def minDisFromLigand(self, atom, ligand):
        distances = self.distancesFromLigand(atom, ligand)
        return min(distances)

if __name__ == '__main__':
    output = 'test_disrmsd.csv'
    recep_dir = '/home/kmk_gmx/Desktop/bioinfo/receptor_dock/'
    lig_dir = '/home/kmk_gmx/Desktop/bioinfo/ligand_dock/'
    cal = DistanceRmsd()
    names = []
    dis_rmsds = []
    for pdbid in os.listdir(recep_dir):
        name = pdbid[:4]
        dis_rmsd = cal.rmsd(recep_dir + pdbid, lig_dir + pdbid)
        names.append(name)
        dis_rmsds.append(dis_rmsd)
        print(pdbid)

    df = pd.DataFrame({'name': names, 'dis_rmsd': dis_rmsds})
    df.to_csv(output)
    


