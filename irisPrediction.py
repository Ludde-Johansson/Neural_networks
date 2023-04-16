import numpy as np
from singleLayerNN import *

def loadFlowerData():
    data = np.genfromtxt("FlowerData/IRIS_twoClasses.csv", delimiter=',')
    data = data[1:, :]
    np.random.seed(0)
    training, test = data[:80,:], data[80:,:]
    data = {"train": training, "test": test}
    return data
data = loadFlowerData()

train = data["train"]
test = data["test"]

X_train = train[:,0:4]
print(X_train.shape)