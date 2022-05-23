import os

input_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_pdbqt/'
output_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_truncated_pdbqt/'

for i in os.listdir(input_dir):
    if len(i) != len('5jwf_5orh.pdbqt'):
        continue
    f = open(input_dir + i, 'r')
    lines = f.readlines()
    content = []
    for line in lines:
        if line[:4] == 'ATOM' or line[0:6] == 'REMARK':
            content.append(line)
    fout = open(output_dir + i, 'w+')
    fout.writelines(content)
