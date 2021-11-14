import os

conf_path = '../v2020_dock_conf/'
vina_path = './vina'
count = 0

for conf in os.listdir(conf_path):
    print('\r' + str(count), end='')
    os.system(vina_path + ' --config ' + conf_path+conf)
    count += 1
