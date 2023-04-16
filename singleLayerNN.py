import numpy as np
import sklearn

import numpy as np

def getNumberOfInputs(X):
    return np.shape(X)[0]

def getNumberOfTrainingExamples(X):
    return np.shape(X)[1]

def initializeLayers(numberOfInputs, nodesHiddenLayer, nodesOutputLayer):
    W1 = np.random.randn(nodesHiddenLayer, numberOfInputs)
    b1 = np.zeros((nodesHiddenLayer, 1))
    W2 = np.random.randn(nodesOutputLayer, nodesHiddenLayer)
    b2 = np.zeros((nodesOutputLayer, 1))
    params = {"W1":W1, "W2": W2, "b1":b1, "b2": b2}
    
    return params
    
def relu(z):
    return np.max(z, 0)

def sigmoid(x):
    y = np.where(x >= 0, 1 / (1+ np.exp(-x)), np.exp(x) / (np.exp(x) + 1))
    return y

def forwardPass(X, params):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "Z2": Z2, "A1": A1, "A2": A2}
    return cache

def backwardPass(X, Y, numberOfTrainingExamples, cache, params):
    W2 = params["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    m = numberOfTrainingExamples

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, keepdims=True, axis=1)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, keepdims=True, axis=1)

    gradients = {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}
    return gradients

def updateParameters(params, gradients, learningRate):
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]
    dW1 = gradients["dW1"]
    dW2 = gradients["dW2"]
    db1 = gradients["db1"]
    db2 = gradients["db2"]
    
    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1
    W2 = W2 - learningRate * dW2
    b2 = b2 - learningRate * db2

#    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#    return parameters

def trainModel(X, Y):
    numberOfTrainingExamples = getNumberOfTrainingExamples(X)
    numberOfInputs = getNumberOfInputs(X)
    parameters = initializeLayers(numberOfInputs, nodesHiddenLayer= 5)
    cache = forwardPass(X, parameters)
    gradients = backwardPass(X, Y, numberOfTrainingExamples, cache, parameters)
    updateParameters(parameters, gradients, learningRate= 0.01)

    return parameters

def predict(X, params):
    cache = forwardPass(X, params)
    Yhat = cache["A2"]
    return Yhat

