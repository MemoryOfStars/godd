import os

command = "../autodock_vina_1_1_2_linux_x86/bin/vina_split --input "
docking_path = "/data/bilab/kaku/testDockings/"
target_path = " /data/bilab/kaku/test_dockings_splited/"

for pdbqt in os.listdir(docking_path):
    if len(pdbqt) == 10:
        os.system(command + docking_path + pdbqt)
        os.system("mv "+docking_path+pdbqt + target_path)
    