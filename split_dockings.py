import os

command = "../autodock_vina_1_1_2_linux_x86/bin/vina_split --input "
docking_path = "/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/split_docking/"
target_path = "/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/docking_output/"

for pdbqt in os.listdir(docking_path):
    if len(pdbqt) == 10:
        os.system(command + docking_path + pdbqt)
        os.system("mv "+docking_path+pdbqt + ' ' + target_path)
    
