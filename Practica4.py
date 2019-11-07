from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integrate
import random as rnd
import scipy.optimize as opt
import displayData as dp

from pandas.io.parsers import read_csv

data = loadmat ("ex4data1.mat")

y = data ["y"]
X = data ["X"]
_lambda = 0.1


muestras = len(X)
x_unos = np.hstack([np.ones((len(X),1)),X])
Y = np.ravel(y)
weights = loadmat('ex4weights.mat')
thetha1,thetha2 = weights['Theta1'], weights['Theta2']
sample = np.random.choice(X.shape[0], 100)
im = X[sample,:]
dp.displayData(im) #esto muestra la tabla con 100 numeros random
dp.displayImage(X[4999]) #esto muestra solo una imagen
plt.show()


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoidDeriv(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def forwardProp():
    

def fun_coste(X,Y, _lambda,K):
    theta1 = np.array(thetha1)
    theta2 = np.array(thetha2)
   # h = sigmoid(np.dot(X,thethas))
    o1 = -(1.0/len(X))
    o2 = np.dot(np.log(h).T,Y)
    o3 = np.dot(np.log(1 - h).T,(1 - Y))
    o4 = (_lambda/(2*len(X)))*np.sum(theta1[1,:]**2)*np.sum(theta2[1,:]**2)
    return o1 * (o2 + o3) + o4
