import os
import numpy as np
import math
'''
parameter: 
naturalFile:真实分子的pdbqt文件路径
dockingFile：docking的pdbqt文件路径
'''

class RMSDCalculator():
    def __init__(self):
        pass

    def calculateRMSD(self, naturalFile, dockingFile):
        '''
        input: two file path
        output: rmsd
        '''
        # print("RMSDCalculator: ", naturalFile, dockingFile, end=' ')
        oriX = [];oriY = [];oriZ = []
        with open(naturalFile) as natFile:
            for line in natFile.readlines():
                if(line[:4] == 'ATOM'):
                    oriX.append(float(line[31:38].strip()))
                    oriY.append(float(line[39:46].strip()))
                    oriZ.append(float(line[47:54].strip()))
        n = len(oriX)
        oriX = np.array(oriX);oriY = np.array(oriY);oriZ = np.array(oriZ)


        dockX = [];dockY = [];dockZ = []
        with open(dockingFile,'r') as dockFile:
            for line in dockFile.readlines():
                if(line[:4] == 'ATOM'):
                    dockX.append(float(line[31:38].strip()))
                    dockY.append(float(line[39:46].strip()))
                    dockZ.append(float(line[47:54].strip()))
        dockX = np.array(dockX);dockY = np.array(dockY);dockZ = np.array(dockZ)
        rmsd = math.sqrt(((dockX-oriX)**2 + (dockY-oriY)**2 + (dockZ-oriZ)**2).sum()/n)
        # print(rmsd)

        return rmsd

if __name__ == '__main__':
    cal = RMSDCalculator()
