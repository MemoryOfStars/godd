import sys

class Pdb2Fasta():
    def __init__(self):
        self.letters = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
           'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
           'TYR': 'Y', 'VAL': 'V'}

    def pdb2fasta(self, pdb_lines):
        fasta_lines = ''
        prev = '-1'
        for line in pdb_lines:
            toks = line.split()
            if len(toks) < 1: continue
            if toks[0] != 'ATOM': continue
            if toks[5] != prev:
                fasta_lines += ('%c' % self.letters[toks[3]])
            prev = toks[5]
        return fasta_lines

if __name__ == '__main__':
    formatter = Pdb2Fasta()
    pdb_lines = open('./5orh_protein.pdb', 'r')
    print(formatter.pdb2fasta(pdb_lines.readlines()))
