import os

command = "../autodock_vina_1_1_2_linux_x86/bin/vina_split --input "
docking_path = "/data/bilab/kaku/docking/"
target_path = " /home/kmk_gmx/Desktop/bioinfo/dockings/"

for pdbqt in os.listdir(docking_path):
    os.system(command + docking_path+pdbqt)
    os.system("mv "+docking_path+pdbqt + target_path)