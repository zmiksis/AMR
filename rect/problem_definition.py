import math
import numpy as np

def domain():

    # Two point
    # dom = [[-2,2]]

    # Two circle
    dom = [[-2,2],[-2,2]]

    # Two sphere
    # dom = [[-2,2],[-2,2],[-2,2]]

    return dom

def grid_size():

    # Two point
    # N = [40]

    # Two circle
    N = [40,40]

    # Two sphere
    # N = [40,40,40]

    return N

def bndry_order():

    return 3

def gamma():

    return 1

def Hamiltonian(xval):

    # Two point
    # x = xval[0]
    # H = abs(x)

    # Two circle
    x = xval[0]
    y = xval[1]
    H = np.sqrt(x**2 + y**2)

    # Two sphere
    # x = xval[0]
    # y = xval[1]
    # z = xval[2]
    # H = np.sqrt(x**2 + y**2 + z**2)

    return H

def gamma_region(xval):

    # Two point
    # x = xval[0]
    # d11 = abs((x+1) - 0.5)
    # d12 = abs((x+1) + 0.5)
    # d21 = abs((x-np.sqrt(1.5)) - 0.5)
    # d22 = abs((x-np.sqrt(1.5)) + 0.5)
    # d = min(d11,d12,d21,d22)
    # if d < 5e-2:
    #     return True
    # else:
    #     return False

    # Two circle
    x = xval[0]
    y = xval[1]
    d1 = abs(np.sqrt((x+1)**2 + y**2) - 0.5)
    d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2) - 0.5)
    d = min(d1,d2)
    if d < 5e-2:
        return True
    else:
        return False

    # Two sphere
    # x = xval[0]
    # y = xval[1]
    # z = xval[2]
    # d1 = abs(np.sqrt((x+1)**2 + y**2 + z**2) - 0.5)
    # d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2 + z**2) - 0.5)
    # d = min(d1,d2)
    # if d < 5e-2:
    #     return True
    # else:
    #     return False


def f(xval):

    # Two point
    # rhs = 1

    # Two circle
    rhs = 1

    # Two sphere
    # rhs = 1

    return rhs


def exact(xval):

    # Two point
    # x = xval[0]
    # d11 = abs((x+1) - 0.5)
    # d12 = abs((x+1) + 0.5)
    # d21 = abs((x-np.sqrt(1.5)) - 0.5)
    # d22 = abs((x-np.sqrt(1.5)) + 0.5)
    # d = min(d11,d12,d21,d22)

    # Two circle
    x = xval[0]
    y = xval[1]
    d1 = abs(np.sqrt((x+1)**2 + y**2) - 0.5)
    d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2) - 0.5)
    d = min(d1,d2)

    # Two sphere
    # x = xval[0]
    # y = xval[1]
    # z = xval[2]
    # d1 = abs(np.sqrt((x+1)**2 + y**2 + z**2) - 0.5)
    # d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2 + z**2) - 0.5)
    # d = min(d1,d2)

    return d

def alpha():

    return [1, 1, 1]
