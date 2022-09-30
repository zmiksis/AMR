import math
import numpy as np
import Hamiltonian as Hml

def domain():

    # Two point
    # dom = [[-2,2]]

    # Two circle
    # dom = [[-2,2],[-2,2]]

    # Two sphere
    # dom = [[-2,2],[-2,2],[-2,2]]

    # Vase
    dom = [[-0.5,0.5],[0,1]]

    # SFS
    # dom = [[0,1],[0,1]]

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

def Hamiltonian(xpoint, xval):

    # Two point
    # x = xval[0]
    # H = abs(x)

    # Two circle
    # x = xval[0]
    # y = xval[1]
    # H = np.sqrt(x**2 + y**2)

    # Two sphere
    # x = xval[0]
    # y = xval[1]
    # z = xval[2]
    # H = np.sqrt(x**2 + y**2 + z**2)

    # Shape-from-shading
    x = xpoint[0]
    y = xpoint[1]
    p = xval[0]
    q = xval[1]
    H = np.sqrt(p**2 + q**2)


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
    # x = xval[0]
    # y = xval[1]
    # d1 = abs(np.sqrt((x+1)**2 + y**2) - 0.5)
    # d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2) - 0.5)
    # d = min(d1,d2)
    # if d < 5e-2:
    #     return True
    # else:
    #     return False

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

    # Vase
    x = xval[0]
    y = xval[1]
    # xbnd = (54/5)*(-1.1266+y)*(0.323846+y)*(0.685836-1.61301*y+y**2)*(0.055506+0.0824245*y+y*2)
    if math.dist([x,y],[0,0.36621]) < 1e-1:
        return True
    elif abs(0.5-abs(x)) < 1e-1:
        return True
    else:
        return False
    # elif abs(y) < 0.2 or abs(1-y) < 1e-2:
    #     return True
    # elif math.dist([x,y],[0,0]) < 1e-2 or math.dist([x,y],[0,1]) < 1e-2:
    #     return True
    # elif x < xbnd or x > -xbnd:
    #     return True

    # SFS
    # x = xval[0]
    # y = xval[1]
    # if abs(x) < 1e-5 or abs(1-x) < 1e-5 or abs(y) < 1e-5 or abs(1-y) < 1e-5:
    #     return True
    # elif math.dist([x,y],[1/4,1/4]) < 1e-5 or math.dist([x,y],[1/4,3/4]) < 1e-5 \
    #     or math.dist([x,y],[3/4,1/4]) < 1e-5 or math.dist([x,y],[3/4,3/4]) < 1e-5 \
    #     or math.dist([x,y],[1/2,1/2]) < 1e-5:
    #     return True
    # else:
    #     return False

def f(xval, Dx, grid):

    # Two point
    # rhs = 1

    # Two circle
    # rhs = 1

    # Two sphere
    # rhs = 1

    # Shape-from-shading
    x = xval[0]
    y = xval[1]
    node = grid.findNode(xval)
    [Txn,Txp] = Hml.LeftRightI(grid,node,0)
    p = Hml.dT(grid,[Txn,Txp],0)
    [Txn,Txp] = Hml.LeftRightI(grid,node,1)
    q = Hml.dT(grid,[Txn,Txp],1)
    N = np.divide(np.array([p,q,-1]),np.sqrt(p**2+q**2+1))
    S = np.array([0,0,1])
    NS = np.dot(N,S)
    rho = 1
    A = 1
    I = A*rho*NS
    if I == 0: I = 1e-6
    rhs = np.sqrt((1/I**2) - 1)

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
    # x = xval[0]
    # y = xval[1]
    # d1 = abs(np.sqrt((x+1)**2 + y**2) - 0.5)
    # d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2) - 0.5)
    # d = min(d1,d2)

    # Two sphere
    # x = xval[0]
    # y = xval[1]
    # z = xval[2]
    # d1 = abs(np.sqrt((x+1)**2 + y**2 + z**2) - 0.5)
    # d2 = abs(np.sqrt((x-np.sqrt(1.5))**2 + y**2 + z**2) - 0.5)
    # d = min(d1,d2)

    # Vase
    x = xval[0]
    y = xval[1]
    fy = 0.15 - 0.1*y*((6*y+1)**2)*((y-1)**2)*(3*y-2)
    z = fy**2 - x**2
    if z < 0:
        d = 0
    else:
        d = np.sqrt(z)

    # SFS
    # x = xval[0]
    # y = xval[1]
    # d = math.sin(2*math.pi*x)*math.sin(2*math.pi*y)


    return d

def alpha():

    return [1, 1, 1]
