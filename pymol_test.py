import pymol

cmd = pymol.cmd

sele = cmd.select('1a4k', ('1a4k_protein', '1a4k_ligand'))

// 把1a4k重合到1a4r上
align 1a4k_protein, 1a4r_protein
// 把protein的matrix复合到ligand上
matrix_copy 1a4k_protein, 1a4k_ligand