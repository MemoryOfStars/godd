import os

input_dir = '../blast_datas/blast_docking/blast_pdbqt/'
output_dir = '../blast_datas/blast_docking/splited_pdbqt/'

for i in os.listdir(input_dir):
    if len(i) != len('5jwf.pdbqt'):
        continue
    f = open(input_dir + i, 'r')
    lines = f.readlines()
    content = []
    for line in lines:
        if line[:4] == 'ROOT' or line[:7] == 'ENDROOT' or line[:6] == 'BRANCH' or line[:9] == 'ENDBRANCH' or line[:7] == 'TORSDOF':
            continue
        content.append(line)
    fout = open(output_dir + i, 'w+')
    fout.writelines(content)
