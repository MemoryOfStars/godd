import os

v2020Dir = '../refined-set/'
v2019Dir = '../refined_set_v2019/'
v2019s = os.listdir(v2019Dir)
testNames = []
for v2020 in os.listdir(v2020Dir):
    if v2020 not in v2019s:
        testNames.append(v2020)

receptor_dir = '../v2020_receptor_dock/'
ligand_dir = '../v2020_ligand_dock/'
count = 0

for pdb_dir in testNames:
    print(count)
    count += 1
    pdb_fn = v2020Dir+pdb_dir+'/'+pdb_dir+'_protein.pdb'
    mol2_fn = v2020Dir+pdb_dir+'/'+pdb_dir+'_ligand.mol2'
    os.system('obabel -i pdb '+pdb_fn+' -o pdbqt -O '+receptor_dir+pdb_dir+'.pdbqt')
    os.system('obabel -i mol2 '+mol2_fn+' -o pdbqt -O '+ligand_dir+pdb_dir+'.pdbqt')