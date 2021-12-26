import dgl
from dgl.data import DGLDataset
import pandas as pd
import torch
from sklearn.utils import shuffle
from data_filter import DataFilter

"""
一个csv文件和一个batch size
生成一个shuffle过的新的batch size里正负例1：1的新df，从这个df里面取数据
"""

class MyDataset(DGLDataset):
    """
    Parameters
    -------------------------
    raw_dir: str
        Specifying the directory that already stores the input data.
    rmsdBound: if a condition need to be fulfilled in this dataset for negative rows.set to (-1, -1) to disable it. 
    """
    def __init__(self, 
                 csvFile,
                 batchSize,
                 rmsdBound,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        self.csvFile = csvFile
        self.batchSize = batchSize
        self.rmsdBound = rmsdBound
        super(MyDataset, self).__init__(name='docking_classify',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        
    def download(self):
        pass
    #must be implemented
    def process(self):
        df = shuffle(pd.read_csv(self.csvFile))
        graphs = df['file_name']
        labels = df['label']
        size = len(graphs)

        self.dataset = pd.DataFrame(columns=['file_name', 'label'])
        
        posiPos = 0
        negaPos = 0
        datasetCount = 0
        dataFilter = DataFilter()

        while negaPos < size:
            count = 0
            dfBatch = pd.DataFrame(columns=['file_name', 'label'])
            while count < self.batchSize/2:
                if posiPos >= size:
                    posiPos = posiPos%size
                if labels[posiPos] == 1.0:
                    dfBatch = dfBatch.append([{'file_name': graphs[posiPos], 'label': torch.Tensor([labels[posiPos]])}], ignore_index=True)
                    count += 1
                posiPos += 1
            while count < self.batchSize and negaPos < size:
                if labels[negaPos] == 0.0:
                    if self.rmsdBound != (-1, -1):
                        dataID = dataFilter.getDataID(graphs[negaPos])
                        dockingPath = dataFilter.getFilePathByDataID(dataID)
                        if not dataFilter.rmsdInbound(graphs[negaPos], self.rmsdBound):
                            negaPos += 1
                            continue
                    dfBatch = dfBatch.append([{'file_name': graphs[negaPos], 'label': torch.Tensor([labels[negaPos]])}], ignore_index=True)
                    count += 1
                negaPos += 1
            self.dataset = self.dataset.append(shuffle(dfBatch), ignore_index=True)


    #must be implemented
    def __getitem__(self, idx):
        """get one item by index
        
        Parameters
        ---------------
        idx: int
            Item index

        Returns
        ---------------
        (dgl.DGLGraph, Tensor)
        """
        # idx.item():convert torch.Tensor to int
        # print("-----idx.item():", idx.item())
        # print(self.dataset['file_name'][idx.item()])
        graph = dgl.load_graphs(self.dataset['file_name'][idx.item()])[0]
        label = self.dataset['label'][idx.item()]
        return graph[0], label[0].float()

    #must be implemented
    def __len__(self):
        #number of data examples
        return self.dataset.shape[0]
        

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass