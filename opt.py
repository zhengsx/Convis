import numpy as np
import func

def GradientDescent(initPoint, functype, learningrate = 0.1, epoch = 100):
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]
    for i in range(1, epoch):
        lineData[0:2, i] = lineData[0:2, i-1] - learningrate * functype.derv(lineData[0:2, i-1])
        lineData[2, i] = functype.calc(lineData[0:2, i])
    return lineData

def GDMomentum(initPoint, functype, rho = 0.95, learningrate = 0.1, epoch = 100):
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]
    grad, delta = np.zeros((2, epoch)), np.zeros((2, epoch))
    for i in range(1, epoch):
        grad[:, i] = functype.derv(lineData[0:2, i-1])
        delta[:, i] = rho * delta[:, i-1] - learningrate * grad[:, i]
        lineData[0:2, i] = lineData[0:2, i-1] + delta[:, i]
        lineData[2, i] = functype.calc(lineData[0:2, i])
    return lineData

def AdaGrad(initPoint, functype, learningrate = 0.1, epoch = 100):
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]
    grad, sigma = np.zeros((2, epoch)), np.zeros((2, epoch))
    for i in range(1, epoch):
        grad[:, i] = functype.derv(lineData[0:2, i-1])
        sigma[:, i] = sigma[:, i-1] + grad[:, i]**2
        delta = - learningrate * grad[:, i] / np.sqrt(sigma[:, i])
        lineData[0:2, i] = lineData[0:2, i-1] + delta
        lineData[2, i] = functype.calc(lineData[0:2, i])
    return lineData

def AdaDelta(initPoint, functype, rho = 0.95, e= 1e-6, epoch = 100):
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]
    g, s = np.zeros((2, epoch)), np.zeros((2, epoch))
    for i in range(1, epoch):
        grad = functype.derv(lineData[0:2, i-1])
        g[:, i] = (1 - rho) * grad**2 + rho * g[:, i-1]
        delta = np.sqrt(s[:, i-1] + e) / np.sqrt(g[:, i] + e) * (-grad)
        s[:, i] = (1 - rho) * delta**2 + rho * s[:, i-1]
        lineData[0:2, i] = lineData[0:2, i-1] + delta
        lineData[2, i] = functype.calc(lineData[0:2, i])
    return  lineData
