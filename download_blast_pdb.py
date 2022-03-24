from Bio.Blast import NCBIXML
from Bio.PDB import PDBList
import os
import re
import pandas as pd

blast_result_dir = '../blast_result/'
blast_pdb_dir = '../blast_pdb/'
identity_thresh = 0.90

pdbl = PDBList()
pattern = re.compile(r'pdb\|(\w+)\|')

source_pdb = []
blast_pdb = []

blast_results = os.listdir(blast_result_dir)
for fname in blast_results:
    result_handle = open(blast_result_dir + fname)
    try:
        blast_record = NCBIXML.read(result_handle)
    except:
        continue
    for align in blast_record.alignments:
        for hsp in align.hsps:
            print('title: %s, identity: %f' % (align.title, hsp.identities/align.length))
            if hsp.identities/align.length > identity_thresh:
                result = pattern.findall(align.title)
                if len(result) < 1:
                    continue
                try:
                    pdbl.retrieve_pdb_file(result[0].lower(), pdir=blast_pdb_dir, file_format='pdb')
                    source_pdb.append(fname[:4])
                    blast_pdb.append(result[0].lower())
                except:
                    continue

df = pd.DataFrame({"source": source_pdb, "blast": blast_pdb})
df.to_csv('./source2blast.csv')
