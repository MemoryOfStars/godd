import os
import re

aligned_pdbqt_path = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_truncated_pdbqt/'
dock_output_path = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_pdbqt_dock/'
source_confs = '/home/kmk_gmx/Desktop/bioinfo/dock_conf/'
conf_output_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_dock_confs/'

for aligned in os.listdir(aligned_pdbqt_path):
    source_id = aligned[5:9]
    source_conf = open(source_confs + source_id + '.txt', 'r').readlines()
    aligned_conf = open(conf_output_dir + aligned[:-6] + '.txt', 'w+')
    source_conf[0] = 'receptor = ' + aligned_pdbqt_path + aligned + '\n'
    xs = 50 #float(source_conf[5].strip()[9:]) * 2
    ys = 50 #float(source_conf[6].strip()[9:]) * 2
    zs = 50 #float(source_conf[7].strip()[9:]) * 2
    source_conf[5] = 'size_x = ' + str(xs) + '\n'
    source_conf[6] = 'size_y = ' + str(ys) + '\n'
    source_conf[7] = 'size_z = ' + str(zs) + '\n'
    source_conf[11] = 'out = ' + dock_output_path + aligned
    aligned_conf.writelines(source_conf)
