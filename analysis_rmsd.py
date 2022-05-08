import os
import pymol
import pandas as pd

cmd = pymol.cmd

# base_dir是最原始的pdb文件
recep_dir = '/home/kmk_gmx/Desktop/bioinfo/receptor_dock/'
ligand_dir = '/home/kmk_gmx/Desktop/bioinfo/ligand_dock/'
blast_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/blast_pdbqt/'
dockings_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/splited_dockings/'
aligned_dock_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_docks/'

source2blast = pd.read_csv('./source2blast.csv')
blastDict = {}
for i, item in source2blast.iterrows():
    blastDict[item['blast']] = item['source']

dock_names = []
dock_rmsds = []
for dock_name in os.listdir(dockings_dir):
    blast_id = dock_name[:4]
    source_id = blastDict[blast_id]
    recep_pymol = recep_dir + source_id + '.pdbqt'
    lig_pymol = ligand_dir + source_id + '.pdbqt'
    blast_pymol = blast_dir + blast_id + '.pdbqt'
    dock_pymol = dockings_dir + dock_name
    cmd.load(recep_pymol, 'recep')
    cmd.load(lig_pymol, 'lig')
    cmd.load(blast_pymol, 'blast')
    cmd.load(dock_pymol, 'dock')
    cmd.align('blast', 'recep')
    cmd.matrix_copy('blast', 'dock')
    aligned_name = aligned_dock_dir + dock_name + '.pdb'
    cmd.save(aligned_name, 'dock')
    cmd.delete('all')
