import numpy as np
import func

def GradientDescent(initPoint, functype, learningrate = 0.01, epoch = 100):
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]
    for i in range(1, epoch):
        lineData[0:2, i] = lineData[0:2, i-1] - learningrate * functype.derv(lineData[0:2, i-1])
        lineData[2, i] = functype.calc(lineData[0:2, i])
    # for i in range(1, epoch):
    #   x = -functype.derv(lineData[0:2, i-1])
    #   learningrate = lineSearch_backtracking(functype, lineData[0:2, i-1], x, alpha=10)
    #   lineData[0:2, i] = lineData[0:2, i-1] + learningrate * x
    #   lineData[2, i] = functype.calc(lineData[0:2, i])
    return lineData

def GDMomentum(initPoint, functype, rho = 0.95, learningrate = 0.01, epoch = 100):
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]
    grad, delta = np.zeros((2, epoch)), np.zeros((2, epoch))
    for i in range(1, epoch):
        grad[:, i] = functype.derv(lineData[0:2, i-1])
        delta[:, i] = rho * delta[:, i-1] - learningrate * grad[:, i]
        lineData[0:2, i] = lineData[0:2, i-1] + delta[:, i]
        lineData[2, i] = functype.calc(lineData[0:2, i])
    return lineData

def AdaGrad(initPoint, functype, learningrate = 0.01, epoch = 100):
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

def NAG(initPoint, functype, rho = 0.95, learningrate = 0.01, epoch = 100):
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]
    grad, delta = np.zeros((2, epoch)), np.zeros((2, epoch))
    for i in range(1, epoch):
        grad[:, i] = functype.derv(lineData[0:2, i-1] + rho * delta[:, i-1])
        delta[:, i] = rho * delta[:, i-1] - learningrate * grad[:, i]
        lineData[0:2, i] = lineData[0:2, i-1] + delta[:, i]
        lineData[2, i] = functype.calc(lineData[0:2, i])
    return lineData

##########################################################################

def BFGS(initPoint, functype, epoch = 100):
    I2 = np.eye(2, 2)
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]
    Hessian, grad = np.matrix(I2 * 0.00001), np.matrix(np.zeros((2, 1)))
    grad = np.matrix(functype.derv(lineData[0:2, 0])).transpose()
    it = epoch
    for i in range(1, epoch):
        grad_old = grad
        d = - Hessian * grad
        learningrate = lineSearch_backtracking(functype, lineData[0:2, i-1], np.array(d.ravel())[0], alpha=1, c1=1e-3)
        s = learningrate * d
        lineData[0:2, i] = lineData[0:2, i-1] + s.transpose()
        lineData[2, i] = functype.calc(lineData[0:2, i])
        grad = np.matrix(functype.derv(lineData[0:2, i])).transpose()
        y = grad - grad_old
        if np.abs(np.sum(y)) < 1e-10:
            it = i
            break
        r = 1.0/ np.sum((y.transpose() * s))
        U = I2 - r * y * s.transpose()
        Hessian = U.transpose() * Hessian * U + r * s * s.transpose()
    if it < epoch:
        lineData = lineData[:, 0:it]
    return lineData

def L_BFGS(initPoint, functype, epoch = 100):
    lineData = np.empty((3, epoch))
    lineData[0:3, 0] = [initPoint[0], initPoint[1], functype.calc(initPoint)]

    return lineData

def lineSearch_backtracking(functype, x0, d, alpha = 1, tau = 0.5, c1 = 1e-3):
    value = functype.calc(x0)
    slope = np.sum(np.matrix(functype.derv(x0)).transpose() * np.matrix(d))
    k = 0
    while functype.calc(x0 + alpha * d) > value + c1 * alpha * slope:
        alpha = tau * alpha
        k = k + 1
        if k > 1000:
            break
    return  alpha





