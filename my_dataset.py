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
                 posCSV,
                 negCSV,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        self.posCSV = posCSV
        self.negCSV = negCSV
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
        df_pos = pd.read_csv(self.posCSV)
        df_neg = pd.read_csv(self.negCSV)
        pos_graphs = df_pos['file_name']
        pos_labels = df_pos['label']
        neg_graphs = df_neg['file_name']
        neg_labels = df_neg['label']

        self.graph_dataset = []
        self.graph_labels = []
        #negative graphs are more
        for i in range(len(neg_graphs)):
            self.graph_dataset.append(pos_graphs[i%len(pos_graphs)])
            self.graph_dataset.append(neg_graphs[i])
            self.graph_labels.append(torch.Tensor([1])) #positive
            self.graph_labels.append(torch.Tensor([0])) #negative
            
        self.df_dataset = pd.DataFrame({'file_name':self.graph_dataset, 'label':self.graph_labels})
        self.df_dataset = shuffle(self.df_dataset)
        #for i in range(len())

    
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
        graph = dgl.load_graphs(self.df_dataset['file_name'][idx.item()])[0] #idx.item():convert torch.Tensor to int
        #print(self.df_dataset['file_name'][idx.item()])
        label = self.df_dataset['label'][idx.item()]
        return graph[0], label[0].float()

    #must be implemented
    def __len__(self):
        #number of data examples
        return self.df_dataset.shape[0]
        

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass