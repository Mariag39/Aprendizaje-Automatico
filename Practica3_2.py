from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integrate
import random as rnd
import scipy.optimize as opt

from pandas.io.parsers import read_csv

data = loadmat ("ex3data1.mat")

y = data ["y"]
X = data ["X"]
_lambda = 0.1


muestras = len(X)
x_unos = np.hstack([np.ones((len(X),1)),X])
Y = np.ravel(y)
weights = loadmat('ex3weights.mat')
thetha1,thetha2 = weights['Theta1'], weights['Theta2']

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

aux = sigmoid(x_unos.dot(thetha1.T))
aux = np.hstack([np.ones((len(aux),1)), aux])


results = sigmoid(aux.dot(thetha2.T))

prediccion = results.argmax(axis = 1) + 1

z = (prediccion == Y) * 1
probabilidad = sum(z)/len(Y)
print("Probabilidad de acierto: ", probabilidad*100,"%")



