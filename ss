DGLError: Expect number of features to match number of nodes (len(u)). Got 111 and 1596 instead.


8A距离改为ligand的min和max坐标之间的关系


bug说明：DGLGraph会根据edge中节点的序号来自动扩充node数。将其改为自增id而非pdb文件中的行号