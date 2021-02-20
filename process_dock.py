import os

conf_path = '../dock_conf/'
vina_path = './vina'

for conf in os.listdir(conf_path):
    os.system(vina_path + ' --config ' + conf_path+conf)
