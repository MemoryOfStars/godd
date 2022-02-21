from Bio.Blast import NCBIWWW
from Bio import SeqIO
import os

refined_set = '../refined-set/'
result_dir = '../blast_result/'
for pdb_id in os.listdir(refined_set):
    with open(refined_set + pdb_id + '/' + pdb_id + '_protein.pdb') as handle:
        print('--------' + pdb_id)
        sequence = next(SeqIO.parse(handle, "pdb-atom"))
        result_handle = NCBIWWW.qblast("blastp", "pdb", sequence.seq)
        save_file = open(result_dir + pdb_id + ".xml", "w")
        try:
            save_file.write(result_handle.read())
        except:
            os.system('echo "%s\n" > a.log' % pdb_id)
        save_file.close()
        result_handle.close()
