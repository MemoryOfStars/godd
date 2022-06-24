import os
import numpy as np
import pandas as pd
import json
import math
from alignment import AlignBlast
import threading

PDBQT_SERIAL_INDEX = 1
PDBQT_ATOM_NAME_INDEX = 2
PDBQT_RESNAME_INDEX = 4
PDBQT_CHAINID_INDEX = 5
PDBQT_SEQID_INDEX = 6
PDBQT_X_POS_INDEX = 8
PDBQT_Y_POS_INDEX = 9
PDBQT_Z_POS_INDEX = 10
PDBQT_TYPE_INDEX  = 14
ELEMENT_TYPE = ['HD', 'C', 'A', 'N', 'NA', 'OA', 'F', 'MG', 'P', 'S', 'CL', 'CA', 'MN', 'FE', 'ZN', 'BR', 'I']
ResMap = {'GLY':'G', 'ALA':'A', 'VAL':'V', 'LEU':'L', 'ILE':'I',
          'PHE':'F', 'TRP':'W', 'TYR':'Y', 'ASP':'D', 'ASN':'N',
          'GLU':'E', 'LYS':'K', 'GLN':'Q', 'MET':'M', 'SER':'S',
          'THR':'T', 'CYS':'C', 'PRO':'P', 'HIS':'H', 'ARG':'R'}

class AlignPairwiseDisRmsd():
    def __init__(self):
        self.alignBlast = AlignBlast()
    def build_crystal(self, recep, lig):
        self.recep = open(recep, 'r').readlines()
        self.lig = open(lig, 'r').readlines()
        recep_data = self.extractAtomLines(self.recep)
        lig_data = self.extractAtomLines(self.lig)

        dis_rmsd = 0
        pair_count = 0
        structure = {}
        for rrow in recep_data:
            for lrow in lig_data:
                dis = self.distanceIn2Atoms(rrow, lrow)
                if dis > 5:
                    continue
                if lrow[PDBQT_SERIAL_INDEX] not in structure:
                    structure[lrow[PDBQT_SERIAL_INDEX]] = [rrow]
                else:
                    structure[lrow[PDBQT_SERIAL_INDEX]].append(rrow)
                dis_rmsd += dis**2
                pair_count += 1
        if pair_count == 0:
            return -1
        return structure, math.sqrt(dis_rmsd/pair_count)

    def RecentResAtom(self, atomline, ResAtoms):
        atomName = atomline[PDBQT_ATOM_NAME_INDEX]
        res = None
        minDis = 9999999.0

        for atom in ResAtoms:
            if atom[PDBQT_ATOM_NAME_INDEX] != atomName:
                continue
            
            if minDis > self.distanceIn2Atoms(atomline, atom):
                minDis = self.distanceIn2Atoms(atomline, atom)
                res = atom
        return res



    def pairwise_disrmsd(self, structure, recep, blast, lig, dock):
        if len(structure) == 0:
            return -1

        self.recep = open(recep, 'r').readlines()
        self.blast = open(blast, 'r').readlines()
        self.lig = open(lig, 'r').readlines()
        self.dock = open(dock, 'r').readlines()
        recep_data = self.extractAtomLines(self.recep)
        blast_data = self.extractAtomLines(self.blast)
        lig_data = self.extractAtomLines(self.lig)
        dock_data = self.extractAtomLines(self.dock)
        align, recepSeqData, blastSeqData = self.alignBlast.align(recep_data, blast_data)

        if len(lig_data) != len(dock_data):
            print('ligand length not compatible ' + lig, dock)
        
        dis_rmsd = 0
        pair_count = 0
        lost_pairs = 0
        for lrow in dock_data:
            if lrow[PDBQT_SERIAL_INDEX] not in structure:
                continue
            for rrow in structure[lrow[PDBQT_SERIAL_INDEX]]:
                if int(rrow[PDBQT_SEQID_INDEX]) not in align:
                    lost_pairs += 1
                    continue
                seqId = align[int(rrow[PDBQT_SEQID_INDEX])]
                resData = blastSeqData[seqId]
                resAtom = self.RecentResAtom(rrow, resData)
                if resAtom == None:
                    continue

                dis_blast = self.distanceIn2Atoms(resAtom, lrow)
                dis_source = self.distanceIn2Atoms(rrow, lrow)
                dis_rmsd += (dis_blast-dis_source)**2
                pair_count += 1
        if pair_count == 0:
            return 0, lost_pairs
        return math.sqrt(dis_rmsd/pair_count), lost_pairs


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

def main(num):
    output = 'new_aligned_blast_pairwise_disrmsd%d.csv' % num
    lig_dir = '/home/kmk_gmx/Desktop/bioinfo/ligand_dock/'
    recep_dir = '/home/kmk_gmx/Desktop/bioinfo/receptor_dock/'
    blast_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_truncated_pdbqt/'
    dock_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_pdbqt_dock/'
    cal = AlignPairwiseDisRmsd()
    names = []
    dis_rmsds = []
    cry_disrmsds = []
    structures = {}
    lost_pairss = []
    count = 0
    for pdbid in os.listdir(dock_dir):
        if len(pdbid) != len('10gs_10gs_ligand_1.pdbqt') or int(pdbid[0]) != num:
            continue
        blast_name = pdbid[:9]
        source_name = blast_name[5:9]
        recep_filepath = recep_dir + source_name + '.pdbqt'
        blast_filepath = blast_dir + blast_name + '.pdbqt'
        lig_filepath = lig_dir + source_name + '.pdbqt'
        if blast_name not in structures:
            structures[blast_name] = cal.build_crystal(recep_filepath, lig_filepath)
        s, cry_disrmsd = structures[blast_name]
        dis_rmsd, lost_pairs = cal.pairwise_disrmsd(s, recep_filepath, blast_filepath, lig_filepath, dock_dir + pdbid)
        names.append(pdbid[:-6])
        dis_rmsds.append(dis_rmsd)
        cry_disrmsds.append(cry_disrmsd)
        lost_pairss.append(lost_pairs)
        print(pdbid[:-6], dis_rmsd, cry_disrmsd, count)
        count += 1

    df = pd.DataFrame({'name': names, 'dis_rmsd': dis_rmsds, 'cry_disrmsd': cry_disrmsds, 'lost_pairs': lost_pairss})
    df.to_csv(output)
  
main(9)
