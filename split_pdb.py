import os
import re

blast_pdb_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/aligned_pdb/'
tar_pdb_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/aligned_truncated_pdb/'

count = 0
for fname in os.listdir(blast_pdb_dir):
    f = open(blast_pdb_dir +  fname, 'r')
    truncated = []
    lines = f.readlines()
    for line in lines:
        if line[:4] == "ATOM":
            truncated.append(line)
    fw = open(tar_pdb_dir + fname, 'w+')
    fw.writelines(truncated)
    print("%d : %s" % (count, fname))
    count += 1
        
        
