import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class IrisData(Dataset):
    def __init__(self, X, Y):
        super(IrisData, self).__init__()
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return X, Y
        
def data_preprocessing(raw_data):
    data = raw_data[["sepal.length","sepal.width","petal.length","petal.width"]]
    target = raw_data[["class"]]

    # Data preprocessing
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Target preprocessing
    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    return data, target