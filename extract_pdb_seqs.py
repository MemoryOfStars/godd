from Bio import PDB
import os

pdbdir = '../refined-set/'
tardir = '../refined_set_seqs/'
parser = PDB.PDBParser()

for pdbid in os.listdir(pdbdir):
    pdbfile = pdbdir + pdbid + '/' + pdbid + '_protein.pdb'
    tar_f = open(tardir + pdbid, 'w+')
    structure = parser.get_structure(pdbid, pdbfile)
    ppb = PDB.PPBuilder()
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq = pp.get_sequence()
        break
    tar_f.write(str(seq))
    tar_f.close()


 
