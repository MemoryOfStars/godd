import os

#v2020Dir = '../refined-set/'
#v2019Dir = '../refined_set_v2019/'
#v2019s = os.listdir(v2019Dir)
#input_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/aligned_truncated_pdb/'
#output_dir = '/home/kmk_gmx/Desktop/bioinfo/blast_datas/blast_docking/aligned_pdbqt/'
input_dir = '/home/kmk_gmx/Desktop/bioinfo/refined-set/'
output_dir = '/home/kmk_gmx/Desktop/bioinfo/refined_set_smiles/'
'''
testNames = []
for v2020 in os.listdir(v2020Dir):
    if v2020 not in v2019s:
        testNames.append(v2020)
'''
#receptor_dir = '../v2020_receptor_dock/'
#ligand_dir = '../v2020_ligand_dock/'
count = 0

for pdb_name in os.listdir(input_dir):
    print(count)
    count += 1
    #pdb_fn = v2020Dir+pdb_dir+'/'+pdb_dir+'_protein.pdb'
    #mol2_fn = v2020Dir+pdb_dir+'/'+pdb_dir+'_ligand.mol2'
    #pdb_id = pdb_name[:-4]
    pdb_id = pdb_name
    input_fn = input_dir + pdb_name + '/' + pdb_name + '_ligand.mol2'
    output_fn = output_dir + pdb_id + '_ligand' + '.smi'
    os.system('obabel -i mol2 '+input_fn+' -o smi -O ' + output_fn)
    #os.system('obabel -i mol2 '+mol2_fn+' -o pdbqt -O '+ligand_dir+pdb_dir+'.pdbqt')
