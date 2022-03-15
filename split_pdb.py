from Bio.PDBParser import PDBParser
import os
import re

blast_pdb_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_pdb/'
blast_receptor_dir = ''
blast_ligand_dir = ''
p = PDBParser(PERMISSIVE=1)
pattern = re.compile(r'pdb(\w+)\.ent')

for fname in os.listdir(blast_pdb_dir):
    result = pattern.findall(fname)
    if len(result) = 0:
        continue
    pdb_id = result[0]
    structure = p.get_structure(pdb_id, blast_pdb_dir + fname)
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()
                het_field = res_id[0]
                if het_field[0] == "H":


