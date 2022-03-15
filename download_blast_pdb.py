from Bio.Blast import NCBIXML
from Bio.PDB import PDBList
import os
import re

blast_result_dir = '../blast_result/'
blast_pdb_dir = '../blast_pdb/'
identity_thresh = 0.90

pdbl = PDBList()
pattern = re.compile(r'pdb\|(\w+)\|')

blast_results = os.listdir(blast_result_dir)
for fname in blast_results:
    result_handle = open(blast_result_dir + fname)
    blast_record = NCBIXML.read(result_handle)
    for align in blast_record.alignments:
        for hsp in align.hsps:
            #print('title: %s, identity: %f' % (align.title, hsp.identities/align.length))
            if hsp.identities/align.length > identity_thresh:
                result = pattern.findall(align.title)
                if len(result) < 1:
                    continue
                pdbl.retrieve_pdb_file(result[0].lower(), pdir=blast_pdb_dir, file_format='pdb')
