'''
对于某些数据需要满足RMSD在一定范围内的需求

比如设定test集时需要指定一个RMSD的范围以保证满足条件
'''

import os
import numpy
from calculate_rmsd import RMSDCalculator

class DataFilter:
    def __init__(self):
        dockingPaths = ['/data/bilab/kaku/dockings/', '/data/bilab/kaku/test_dockings_splited/']
        naturalPaths = ['/home/kmk_gmx/Desktop/bioinfo/ligand_dock/', '/home/kmk_gmx/Desktop/bioinfo/v2020_ligand_dock/']

        self.dockingFiles = []
        self.naturalFiles = []
        for path in dockingPaths:
            self.dockingFiles += [path + i for i in os.listdir(path)]
        for path in naturalPaths:
            self.naturalFiles += [path + i for i in os.listdir(path)]


    def getDataID(self, fileName):
        '''
        transform name of this docking data to the crystal data name
        '''
        dotIndex = fileName.rfind('.') #去除文件后缀
        if dotIndex != -1:
            fileName = fileName[:dotIndex]
        if '/' in fileName:
            fileName = fileName[fileName.rfind('/')+1:]

        # print("getDataID", fileName)
        if 'ligand' in fileName:
            start = fileName.find('ligand')-5
            end = fileName.find('ligand')+8
            return fileName[start:end]
        else:
            return fileName[-4:]

    def getFilePathByDataID(self, dataID):
        """
        dataID应该能够自动识别出来
        dataID指的就是那个标识，positive比如是5orh，negative比如是5orh_ligand_1
        """
        filePaths = []
        if 'ligand' in dataID:
            filePaths = self.dockingFiles
        else:
            filePaths = self.naturalFiles

        for fname in filePaths:
            if dataID in fname:
                return fname
        # print("Unable to find file:" + dataID)
        return ""

    def getNaturalPathByDataID(self, dataID):
        if 'ligand' in dataID:
            dataID = dataID[:4]
        for fname in self.naturalFiles:
            if dataID in fname:
                return fname
        # print("getNaturalPathByDataID: Can't find positive data:" + dataID)
        return ""


    def rmsdInbound(self, negativeGraphFileName, rmsdBound):
        """
        negativeGraphFileName是一个DGLGraph文件
        rmsdBound是一个长度为2的tuple
        """
        if type(rmsdBound) != type((0,0)) or len(rmsdBound) != len((0,0)):
            print("rmsdBound must be a tuple(size of 2)")
            return False
        # print("data_filter.py[line 76]: rmsdInbound: ", negativeGraphFileName)
        negativeGraphFileName = negativeGraphFileName[negativeGraphFileName.rfind('/')+1:]
        # print("data_filter.py[line 78]: rmsdInbound: ", negativeGraphFileName)
        if 'ligand' not in negativeGraphFileName:
            print("rmsdInbound: fileName must have substring('ligand') !!")
            return False
        lowerBound = min(rmsdBound)
        upperBound = max(rmsdBound)
        
        dataId = self.getDataID(negativeGraphFileName)
        negativeFileName = self.getFilePathByDataID(dataId)

        positiveFileName = self.getNaturalPathByDataID(dataId)
        if positiveFileName == "":
            print("Positive data:" + dataId + "not in natrual paths")
            return False

        rmsdCal = RMSDCalculator()
        rmsd = rmsdCal.calculateRMSD(positiveFileName, negativeFileName)

        return rmsd >= lowerBound and rmsd <= upperBound



