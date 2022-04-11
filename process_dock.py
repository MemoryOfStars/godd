import os

conf_path = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/docking_confs/'
vina_path = './vina'
count = 0

for conf in os.listdir(conf_path):
    print('\r' + str(count), end='')
    os.system(vina_path + ' --config ' + conf_path+conf)
    count += 1
