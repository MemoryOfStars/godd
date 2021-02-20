# 将pdb文件转移到另一个文件夹使用racoon进行批量转换格式

import os

for pdb_dir in os.listdir('../refined-set/'):
    os.system('cp ../refined-set/'+pdb_dir+'/'+pdb_dir+'_protein.pdb '+'../to_be_converted/')