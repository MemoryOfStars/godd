import os

smi_dir = '/home/kmk_gmx/Desktop/bioinfo/refined_set_smiles/'
seq_dir = '/home/kmk_gmx/Desktop/bioinfo/refined_set_seqs/'
output_dir = '/home/kmk_gmx/Desktop/bioinfo/GanDTI/refined_set_data'
output = open(output_dir, 'w+')

for i in os.listdir(smi_dir):
    smi = open(smi_dir + i, 'r').readline().strip().split()[0]
    if i[:4] not in os.listdir(seq_dir):
        continue
    seq = open(seq_dir + i[:4], 'r').readline()
    output.write(smi + ' ' + seq + ' ' + ' 1 ' + i[:4] + '\n')

output.close()
