from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class NNet(Model):
    def __init__(self):
        super(NNet, self).__init__()
        self.fc1 = Dense(3, activation="softmax", input_shape=(4,))

    def call(self, x):
        return self.fc1(x)

