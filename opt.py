import numpy as np
import func

def GradientDescent(initPoint, functype, learningrate = 0.1, epoch = 100):
    x, y, z= np.empty(epoch), np.empty(epoch), np.empty(epoch)
    x[0], y[0], z[0] = initPoint[0], initPoint[1], functype.calc(initPoint[0], initPoint[1])
    for i in range(1, epoch):
        x[i] = x[i - 1] - learningrate * functype.derv(x[i - 1], y[i - 1])[0]
        y[i] = y[i - 1] - learningrate * functype.derv(x[i - 1], y[i - 1])[1]
        z[i] = functype.calc(x[i], y[i])

    lineData = np.empty((3, epoch))
    lineData[0,:],lineData[1,:],lineData[2,:] = x, y, z
    return lineData

def GDMomentum(initPoint, functype, rho = 0.95, learningrate = 0.1, epoch = 100):
    x, y, z= np.empty(epoch), np.empty(epoch), np.empty(epoch)
    x[0], y[0], z[0] = initPoint[0], initPoint[1], functype.calc(initPoint[0], initPoint[1])
    grad, delta = np.zeros((2, epoch)), np.zeros((2, epoch))
    for i in range(1, epoch):
        grad[:, i] = functype.derv(x[i-1], y[i-1])
        delta[:, i] = rho * delta[:, i-1] - learningrate * grad[:, i]
        x[i] = x[i-1] + delta[0, i]
        y[i] = y[i-1] + delta[1, i]
        z[i] = functype.calc(x[i], y[i])
    lineData = np.empty((3, epoch))
    lineData[0,:],lineData[1,:],lineData[2,:] = x, y, z
    return lineData

def AdaGrad(initPoint, functype, epoch = 100):

    lineData = np.empty((3, epoch))
    return lineData

def AdaDelta(initPoint, functype, rho = 0.95, e= 1e-6, epoch = 100):
    x, y, z= np.empty(epoch), np.empty(epoch), np.empty(epoch)
    x[0], y[0], z[0] = initPoint[0], initPoint[1], functype.calc(initPoint[0], initPoint[1])
    g, s = np.zeros((2, epoch)), np.zeros((2, epoch))
    for i in range(1, epoch):
        grad = functype.derv(x[i-1], y[i-1])
        g[:, i] = (1 - rho) * grad**2 + rho * g[:, i-1]
        delta = np.sqrt(s[:, i-1] + e) / np.sqrt(g[:, i] + e) * (-grad)
        s[:, i] = (1 - rho) * delta**2 + rho * s[:, i-1]
        x[i] = x[i-1] + delta[0]
        y[i] = y[i-1] + delta[1]
        z[i] = functype.calc(x[i], y[i])
    lineData = np.empty((3, epoch))
    lineData[0,:],lineData[1,:],lineData[2,:] = x, y, z
    return  lineData
