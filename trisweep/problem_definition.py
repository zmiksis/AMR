import math
import numpy as np

def f(x, y):

    # Two-circle problem
    rhs = 1

    # Shape-from-shading
    # rhs1 = math.cos(2*math.pi*x)*math.sin(2*math.pi*y)
    # rhs2 = math.sin(2*math.pi*x)*math.cos(2*math.pi*y)
    # rhs = 2*math.pi*np.sqrt(rhs1**2 + rhs2**2)

    return rhs

def gamma_region(x, y):

    # Two-circle problem
    d1 = abs(np.sqrt((x+1)**2 + y**2) - 0.5)
    d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2) - 0.5)
    d = min(d1,d2)

    if d < 5e-2:
        return True
    else:
        return False

    # Shape-from-shading
    # Gx = np.array([1/4, 3/4, 1/4, 3/4, 1/2])
    # Gy = np.array([1/4, 3/4, 3/4, 1/4, 1/2])
    # d = np.sqrt(abs(Gx-x)**2 + abs(Gy-y)**2)
    #
    # if min(d) < 5e-2 or x < 5e-2 or x > 1-5e-2 or y < 5e-2 or y > 1-5e-2:
    #     return True
    # else:
    #     return False

def exact(x, y):

    # Two-circle problem
    d1 = abs(np.sqrt((x+1)**2 + y**2) - 0.5)
    d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2) - 0.5)
    d = min(d1,d2)

    return d

    # Shape-from-shading
    # d = np.sin(2*math.pi*x)*np.sin(2*math.pi*y)
    #
    # return d

if __name__ == '__main__':
    print('Main driver not run')
    pass
