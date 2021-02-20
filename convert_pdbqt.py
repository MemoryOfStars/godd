import os

receptor_dir = '../receptor_dock/'
ligand_dir = '../ligand_dock/'
base_dir = '../refined-set/'
count = 0

for pdb_dir in os.listdir(base_dir):
    print(count)
    count += 1
    pdb_fn = base_dir+pdb_dir+'/'+pdb_dir+'_protein.pdb'
    mol2_fn = base_dir+pdb_dir+'/'+pdb_dir+'_ligand.mol2'
    #os.system('obabel -i pdb '+pdb_fn+' -o pdbqt -O '+receptor_dir+pdb_dir+'.pdbqt' + ' -h')
    os.system('obabel -i mol2 '+mol2_fn+' -o pdbqt -O '+ligand_dir+pdb_dir+'.pdbqt')