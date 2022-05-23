import pymol
import pandas as pd
import os

blast_pdb_path = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_pdb/'
source_pdb_path = '/home/kmk_gmx/Desktop/bioinfo/refined-set/'
aligned_path = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/aligned_pdb/'

cmd = pymol.cmd

source2blast = pd.read_csv('./source2blast.csv')
blastDict = {}
for i, item in source2blast.iterrows():
    if item['blast'] not in blastDict:
        blastDict[item['blast']] = [item['source']]
    blastDict[item['blast']].append(item['source'])

def sourcePath(sourceId):
    return source_pdb_path + sourceId + '/' + sourceId + '_protein.pdb'

for pdb_name in os.listdir(blast_pdb_path):
    # pdb5orh.ent
    pdb_id = pdb_name[3:7]
    
    sourceIds = blastDict[pdb_id]
    for sourceId in sourceIds:
        cmd.load(blast_pdb_path + pdb_name, 'blast')
        cmd.load(sourcePath(sourceId), 'source')
        cmd.align('blast', 'source')
        cmd.save('%s%s_%s.pdb' % (aligned_path, pdb_id, sourceId), 'blast')
        print('%s%s_%s.pdb' % (aligned_path, pdb_id, sourceId))
        cmd.delete('all')


