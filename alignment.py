from Bio import pairwise2
from Bio.Seq import Seq

ResMap = {'GLY':'G', 'ALA':'A', 'VAL':'V', 'LEU':'L', 'ILE':'I',
          'PHE':'F', 'TRP':'W', 'TYR':'Y', 'ASP':'D', 'ASN':'N',
          'GLU':'E', 'LYS':'K', 'GLN':'Q', 'MET':'M', 'SER':'S',
          'THR':'T', 'CYS':'C', 'PRO':'P', 'HIS':'H', 'ARG':'R'}
PDBQT_SERIAL_INDEX = 1
PDBQT_ATOM_NAME_INDEX = 2
PDBQT_RESNAME_INDEX = 4
PDBQT_CHAINID_INDEX = 5
PDBQT_SEQID_INDEX = 6
PDBQT_X_POS_INDEX = 8
PDBQT_Y_POS_INDEX = 9
PDBQT_Z_POS_INDEX = 10
PDBQT_TYPE_INDEX  = 14

class AlignBlast:
    def __init__(self):
        pass

    def align(self, atomLines1, atomLines2):
        seq1 = ''
        seq2 = ''
        seqData1 = {}       # SeqNum到其他字段的映射
        seqData2 = {}
        seqIdMap1 = {}      # Sequence序号映射到PDBQT的seqId
        seqIdMap2 = {}
        cursor1 = 0         # Sequence序号，文件顺序
        cursor2 = 0
        alignMap = {}       # seq1到seq2的alignment 映射 SeqId
        for line in atomLines1:
            if line[PDBQT_RESNAME_INDEX] not in ResMap:
                continue
            seqNum = int(line[PDBQT_SEQID_INDEX])
            if seqNum not in seqData1:
                seqIdMap1[cursor1] = seqNum
                cursor1 += 1
                seqData1[seqNum] = [line]
                seq1 += ResMap[line[PDBQT_RESNAME_INDEX]]
                continue
            seqData1[seqNum].append(line)

        for line in atomLines2:
            if line[PDBQT_RESNAME_INDEX] not in ResMap:
                continue
            seqNum = int(line[PDBQT_SEQID_INDEX])
            if seqNum not in seqData2:
                seqIdMap2[cursor2] = seqNum
                cursor2 += 1
                seqData2[seqNum] = [line]
                seq2 += ResMap[line[PDBQT_RESNAME_INDEX]]
                continue
            seqData2[seqNum].append(line)
            
        seqq1 = Seq(seq1)
        seqq2 = Seq(seq2)
        alignment = pairwise2.align.localxx(seq1, seq2)[0]
        print(alignment)
        print(len(seqIdMap1), len(seqIdMap2))

        cursor1 = 0
        cursor2 = 0
        align1 = alignment[0]
        align2 = alignment[1]
        for i in range(len(alignment[0])):
            if align1[i] == '-':
                cursor2 += 1
                continue
            if align2[i] == '-':
                cursor1 += 1
                continue
            if align1[i] != '-' and align2[i] != '-':
                alignMap[seqIdMap1[cursor1]] = seqIdMap2[cursor2] 
                cursor1 += 1
                cursor2 += 1
        print(alignMap)

        return alignMap, seqData1, seqData2
