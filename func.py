import numpy as np

class Domain:
    def __init__(self, minx, maxx, miny, maxy):
        self.minx, self.maxx, self.miny, self.maxy = minx, maxx, miny, maxy

def meshxy(domain, interval = 0.25):
    x = np.arange(domain.minx, domain.maxx, interval)
    y = np.arange(domain.miny, domain.maxy, interval)
    x,y = np.meshgrid(x, y)
    z = 0
    map = np.array([x, y, z])
    return map

def fGoldsteinPrice(point):
    x, y = point[0], point[1]
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))\
                * (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

def dGoldsteinPrice(point):
    x, y = point[0], point[1]
    dx = (2*(x+y+1)*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)+(x+y+1)**2*(-14+6*x+6*y))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))+\
        (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(4*(2*x-3*y)*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)+(2*x-3*y)**2*(-32+24*x-36*y))
    dy = (2*(x+y+1)*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)+(x+y+1)**2*(-14+6*x+6*y))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))+\
        (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(-6*(2*x-3*y)*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)+(2*x-3*y)**2*(48-36*x+54*y))
    derv = np.array([dx, dy])
    return derv

def GoldsteinPrice(interval = 0.1):
    domain = Domain(-2.0, 2.0, -2.0, 2.0)
    map = meshxy(domain, interval)
    map[2] = fGoldsteinPrice(map[0:2])
    return map

def fRosenbroke(point):
    x, y = point[0], point[1]
    z = 100 * (y - x**2)**2 + (x - 1)**2
    return z

def dRosenbroke(point):
    x, y = point[0], point[1]
    dx = 400 * x**3 - 400 * x * y + 2 * x - 2
    dy = 200 * (y - x**2)
    derv = np.array([dx, dy])
    return derv

def Rosenbroke(interval = 0.25):
    domain = Domain(-4.0, 4.0, -5.0, 10.0)
    map = meshxy(domain, interval)
    map[2] = fRosenbroke(map[0:2])
    return map

fCategory = {
    "GoldsteinPrice": (fGoldsteinPrice, dGoldsteinPrice, GoldsteinPrice, Domain(-2.0, 2.0, -2.0, 2.0)),
    "Rosenbroke": (fRosenbroke, dRosenbroke, Rosenbroke, Domain(-4.0, 4.0, -5.0, 10.0))
}

class functype:
    def __init__(self, f):
        self.calc = fCategory.get(f, "no this function type.")[0]
        self.derv = fCategory.get(f, "no this function type.")[1]
        self.func = fCategory.get(f, "no this function type.")[2]
        self.domain = fCategory.get(f, "no this function type.")[3]

if __name__ == "__main__":
    print(0)
