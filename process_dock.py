import os
import threading

conf_path = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_dock_confs/'
vina_path = './vina'

confs = os.listdir(conf_path)

def process(num):
    if num >= 10 or num <= 0:
        return
    for conf in confs:
        if conf[0] != str(num):
            continue
        os.system(vina_path + ' --config ' + conf_path+conf)

threads = []
for i in range(1, 10):
    threads.append(threading.Thread(target=process, args=[i]))

for thread in threads:
    thread.start()
