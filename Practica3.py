from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integrate
import random as rnd

from pandas.io.parsers import read_csv

data = loadmat ("ex3data1.mat")

y = data ["y"]
X = data ["X"]
_lambda = 0.1


muestras = len(X)
x_unos = np.hstack([np.ones((len(X),1)),X])
thethas = np.zeros(len(x_unos[0]))
Y = np.ravel(y)

sample = np.random.choice(X.shape[0], 10)
plt.imshow(X[sample,:].reshape(-1, 20).T)
plt.axis("off")
plt.show()

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def H(thethas,X):
    return sigmoid(X.dot(thethas))

def fun_coste(thethas,vX,vY,muestras,_lambda):
    h = sigmoid(np.dot(vX,thethas))
    o1 = -(1.0/muestras)
    o2 = np.dot(np.log(h).T,vY)
    o3 = np.dot(np.log(1 - h).T,(1 - vY))
    o4 = (_lambda/(2*muestras))*np.sum(thethas**2)
    return o1 * (o2 + o3) + o4

def gradiente_dos(thethas,vX,vY,muestras,_lambda):
    h = sigmoid(np.dot(vX,thethas))
    o1 = np.dot((1.0/muestras),vX.T)
    o2 = h - vY
    o3 = (_lambda/(muestras)) * thethas
    return np.dot(o1,o2) + o3

print(fun_coste(thethas,x_unos,Y,muestras,_lambda))

