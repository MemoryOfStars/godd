from Bio.PDB import PDBParser
import os
import re

blast_pdb_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_pdb/'
blast_receptor_dir = ''
blast_ligand_dir = ''
p = PDBParser(PERMISSIVE=1)
pattern = re.compile(r'pdb(\w+)\.ent')
tar_pdb_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_docking/pdb_truncated/'

count = 0
for fname in os.listdir(blast_pdb_dir):
    f = open(blast_pdb_dir +  fname, 'r')
    truncated = []
    lines = f.readlines()
    for line in lines:
        if line[:6] == "HETATM":
            continue
        truncated.append(line)
    fw = open(tar_pdb_dir + fname, 'w+')
    fw.writelines(truncated)
    print("%d : %s" % (count, fname))
    count += 1
        
        

