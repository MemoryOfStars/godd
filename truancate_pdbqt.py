import os

pro_pdbqt = '../receptor_dock/'
count = 0

for i in os.listdir(pro_pdbqt):
    pf_content = []
    truncated = []
    print(count)
    count+=1
    with open(pro_pdbqt+i) as pf:
        pf_content = pf.readlines()
    for line in pf_content:
        if line[0:4] == 'ATOM' or line[0:6] == 'REMARK':
            truncated.append(line)
    with open(pro_pdbqt+i, "w") as pf:
        pf.writelines(truncated)