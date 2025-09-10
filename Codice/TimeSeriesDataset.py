import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from PrepareData import prepareData
from init import Test, AT, RC, MinMax,max_len
class TimeSeriesDataset(Dataset):

    def __init__(self, data_path, index=None, columns=None):
        self.df_map,self.max_len = prepareData(data_path,Test)
        self.df_AT,len1=prepareData(data_path,AT)
        self.df_RC,len2=prepareData(data_path,RC)
        self.df_MinMax,len3=prepareData(data_path,MinMax)
        # create a numeric index for the loaded files
        self.df_idx = sorted(self.df_map.keys())
        # select a subset of the input data eventually
        if index is not None:
            self.df_idx = [self.df_idx[i] for i in index]
        #contiene gli indici di riga che mi servono
        self.features = self.df_map[self.df_idx[0]].columns[1:-1]# dalla prima alla penultima
        #self.features = self.df_map[self.df_idx[0]].columns[[1, 6]]#la seconda e la settima
        #self.features = self.df_map[self.df_idx[0]].columns[[1]]#la seconda
        df = self.df_map[self.file_name(1)]
        #print(f"Trovati {len(self.df_map)} file per il dataset Test.")
        #print(f"Trovati {len(self.df_AT)} file per il dataset AT.")
        #print(f"Trovati {len(self.df_RC)} file per il dataset RC.")
        #print(f"Indici dei file nel dataset: {self.df_idx}")
        #print(self.features)
        #print(self.file_name(1))
        #print(self.df_map[self.file_name(1)])
        #print(df.loc[:, self.features].astype(np.float32).to_numpy())
        #print(df.loc[:, df.columns[-1:]].astype(np.float32).to_numpy())

    def __len__(self):
        return len(self.df_map)
    def _max_len(self):
        return self.max_len


    def __getitem__(self, item):
       #print(f"accedo all'elemento con indice: {item}")
        #Accedo al dataset del mio file excell foglio1_scalato
        df = self.df_map[self.file_name(item)]
        dfMinMax = self.df_MinMax[self.file_name(item)]
        min_val = dfMinMax.iloc[0, 1].astype(np.float32).item()
        max_val = dfMinMax.iloc[1, 1].astype(np.float32).item()
        #Prendo solo le feature che considero rilevanti
        features = df.loc[:, self.features].astype(np.float32).to_numpy()
        #print(f"  Features estratte. Shape: {features.shape}")
        #Prendiamon le 2 labels e le inseriamo in una unica tupla
        #Accendo al dataset di At e Rc
        dfAt=self.df_AT[self.file_name(item)]
        dfRc=self.df_RC[self.file_name(item)]
        #Prendiamo le 2 labels e le inseriamo in una unica tupla
        lablesAt=dfAt.iloc[:,0].astype(np.float32).item() # per il tempo
        #lablesAt=dfAt.iloc[:,8].astype(np.float32).item() # per il load
        #prendiamo il load come label
        #lablesPower = dfAt.iloc[:, -1].astype(np.float32),item()
       # print(f" shape: {lablesAt.shape}")
        labelsRc=dfRc.iloc[:,0].astype(np.float32).item()# per il tempo
        #labelsRc=dfRc.iloc[:,8].astype(np.float32).item()# per il load


        current_len = features.shape[0]
        if current_len < max_len:
            padding_needed = max_len - current_len
            padding = torch.zeros((padding_needed, features.shape[1]), dtype=torch.float32)
            padded_features = torch.cat((torch.from_numpy(features), padding), dim=0)
        elif current_len > max_len:
            padded_features = torch.from_numpy(features[:self.max_len])
        else:
            padded_features = torch.from_numpy(features)
        #first_column = df.iloc[:, 0].to_numpy()
        current_file_name = self.file_name(item)

        #print(f"  Labels estratti. Shape: {torch.tensor([lablesAt, labelsRc])}")
        return padded_features, torch.tensor([lablesAt, labelsRc]), min_val, max_val, current_file_name
    def _getLabels(self,item):
        r1,r2,r3=self.__getitem__(item)
        return r2
    def file_name(self, id):
        return self.df_idx[id]
