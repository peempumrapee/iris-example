import pandas as pd
import numpy as np
import tensorflow as tf

from model import NNet
from preprocess import data_preprocessing
from trainer import train

if __name__ == '__main__':
    # Get dataset
    raw_data = pd.read_csv('iris.csv')
    data, target, _, _ = data_preprocessing(raw_data)
    trainable = tf.data.Dataset.from_tensor_slices((data, target)).shuffle(100)

    # Define model
    model = NNet()
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(data, target, epochs=5)
    train(model, trainable, 5)

