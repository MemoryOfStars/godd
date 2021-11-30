import dgl
from dgl.data import DGLDataset
import pandas as pd
import torch
from sklearn.utils import shuffle


class MyDataset(DGLDataset):
    """
    Parameters
    -------------------------
    raw_dir: str
        Specifying the directory that already stores the input data.
    
    """
    def __init__(self, 
                 csvFile,
                 batchSize,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        self.csvFile = csvFile
        self.batchSize = batchSize
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

        while negaPos < size:
            count = 0
            dfBatch = pd.DataFrame(columns=['file_name', 'label'])
            while count < self.batchSize/2:
                if posiPos >= size:
                    posiPos = posiPos%size
                if labels[posiPos] == 1.0:
                    dfBatch.append([{'file_name': graphs[posiPos], 'label': labels[posiPos]}])
                    count += 1
                posiPos += 1
            while count < self.batchSize and negaPos < size:
                if labels[negaPos] == 0.0:
                    dfBatch.append([{'file_name': graphs[negaPos], 'label': labels[negaPos]}])
                    count += 1
                negaPos += 1
            self.dataset.append(shuffle(dfBatch))


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
        graph = dgl.load_graphs(self.dataset['file_name'][idx.item()])[0]
        label = self.dataset['label'][idx.item()]
        return graph[0], label.float()

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