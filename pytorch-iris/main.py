#%%
import pandas as pd

from preprocess import data_preprocessing, IrisData
from model import NNet, train, eval

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

if __name__ == '__main__':
    raw_data = pd.read_csv('iris.csv')

    # Data preprocessing 
    data, target = data_preprocessing(raw_data)
    iris = IrisData(data, target)
    irisloader = DataLoader(iris, shuffle=True, batch_size=16)

    # Define Model, Loss function and Optimizer
    model = NNet().float()
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Model processing
    train(model, irisloader, criteria, optimizer, 10)
    # eval(model, irisloader, criteria)
