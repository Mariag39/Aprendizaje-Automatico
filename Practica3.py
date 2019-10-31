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

def fun_coste(thethas,vX,vY,_lambda):
    h = sigmoid(np.dot(vX,thethas))
    o1 = -(1.0/len(vX))
    o2 = np.dot(np.log(h).T,vY)
    o3 = np.dot(np.log(1 - h).T,(1 - vY))
    o4 = (_lambda/(2*len(vX)))*np.sum(thethas**2)
    return o1 * (o2 + o3) + o4

def gradiente_dos(thethas,vX,vY,_lambda):
    h = sigmoid(np.dot(vX,thethas))
    o1 = np.dot((1.0/len(vX)),vX.T)
    o2 = h - vY
    o3 = (_lambda/(len(vX))) * thethas
    return np.dot(o1,o2) + o3

def onevsall(x,y,num,reg):
    thethas = np.zeros([num,x.shape[1]])
    for i in range(num):
        if(i == 0):
            aux = 10
        else:
            aux = i

        a = (y == aux) * 1
        thethas[i] = opt.fmin_tnc(fun_coste,thethas[i],gradiente_dos,args=(x,a,reg))[0]
    return thethas



thethas_opt = onevsall(x_unos,Y,10,0.1)
res = H(thethas_opt.T,x_unos)

prediccion = res.argmax(axis = 1)
prediccion[prediccion == 0] = 10

Z = (prediccion == Y) * 1
probabilidad = sum(Z)/len(Y)

print("Probabilidad de acierto: ", probabilidad*100,"%")
