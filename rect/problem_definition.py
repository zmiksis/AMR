import math
import numpy as np

def domain():

    # Two point
    # dom = [[-2,2]]

    # Two circle
    # dom = [[-2,2],[-2,2]]

    # Two sphere
    # dom = [[-2,2],[-2,2],[-2,2]]

    # Vase
    dom = [[-0.5,0.5],[0,1]]

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
    # f = 1
    # alpha, beta, gam = 0, 0, 1
    # x = xpoint[0]
    # y = xpoint[1]
    # xx = xval[0]
    # yy = xval[1]
    #
    # # I = f*(alpha*xx + beta*yy) + gam*(x*xx + y*yy + 1)
    # # I /= np.sqrt(f*f*(xx**2 + yy**2) + (x*xx + y*yy +1)**2)
    # slant = math.acos(gam)
    # tilt = math.atan(beta/(alpha+1e-5))
    # I = math.cos(slant) + xx*math.cos(tilt)*math.sin(slant) + yy*math.sin(tilt)*math.sin(slant)
    # I /= np.sqrt(1 + xx**2 + yy**2)
    # H = I*np.sqrt(f*f*(xx**2 + yy**2) + (x*xx + y*yy +1)**2)
    # H -= (f*alpha + gam*x)*xx
    # H -= (f*beta + gam*y)*yy
    # H -= gam

    x = xpoint[0]
    y = xpoint[1]
    p = xval[0]
    q = xval[1]
    sigma = 0
    wd = 1
    ws = 0
    n = 10
    A = 1 - (0.5*sigma**2)/(sigma**2 + 0.33)
    B = (0.45*sigma**2)/(sigma**2 + 0.09)
    Itmp = 1/np.sqrt(1+p**2+q**2)
    I = wd*(A*Itmp + B - B*Itmp**2) + ws*Itmp**n
    Tk = 0
    Tkn = 1
    while abs(Tk - Tkn) > 1e-8:
        Tkn = Tk
        FT = ws*Tk**n - B*wd*Tk**2 + A*wd*Tk + B*wd - I
        FpT = n*ws*Tk**(n-1) + wd*(A - 2*B*Tk)
        Tk = Tkn - FT/FpT
    T = Tk
    # p0 = p-1e-6
    # q0 = q-1e-6
    # T = (q0-q)*q0/((1+p0**2+q0**2)**(3/2)) + 1/np.sqrt(1+p0**2+q0**2) \
    #     + ((q-q0)**2)*(2*q0**2-p0**2-1)/(2*(1+p0**2+q0**2)**(5/2)) \
    #     + (p-p0)*(3*p0*(q-q0)**2*(1+p0**2-4*q0**2)/(2*(1+p0**2+q0**2)**(7/2)) \
    #                 + 3*p0*(q-q0)*q0/((1+p0**2+q0**2)**(5/2)) \
    #                 - p0/((1+p0**2+q0**2)**(3/2))) \
    #     + ((p-p0)**2)*(3*(q-q0)*q0*(1-4*p0**2+q0**2)/(2*(1+p0**2+q0**2)**(7/2)) \
    #                     + (2*p0**2-q0**2-1)/(2*(1+p0**2+q0**2)**(5/2)) \
    #                     - 3*(q-q0)**2*(4*p0**4-1+3*q0**2+4*q0**4+p0**2*(3-27*q0**2))/(4*(1+p0**2+q0**2)**(9/2)))
    # T = (1/64)*(8*(8-4*(q**2)+3*(q**4)) - 4*(p**2)*(8-12*(q**2)+15*(q**4)) + 3*(p**4)*(8-20*(q**2)+35*(q**4)))
    # T = (1/4)*(4-2*(q**2)+(p**2)*(3*(q**2)-2))
    # if not math.isnan(T): print(T)
    # else: print("Uh-oh!")
    H = np.sqrt(p**2+q**2) - np.sqrt((1/(T**2))-1)


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
    xbnd = (54/5)*(-1.1266+y)*(0.323846+y)*(0.685836-1.61301*y+y**2)*(0.055506+0.0824245*y+y*2)
    if math.dist([x,y],[0,0.36621]) < 1e-2:
        return True
    elif abs(0.5-abs(x)) < 1e-2:
        return True
    elif abs(y) < 0.2 or abs(1-y) < 1e-2:
        return True
    else:
        return False

    # elif math.dist([x,y],[0,0]) < 1e-2 or math.dist([x,y],[0,1]) < 1e-2:
    #     return True
    # elif x < xbnd or x > -xbnd:
    #     return True


def f(xval, Dx):

    # Two point
    # rhs = 1

    # Two circle
    # rhs = 1

    # Two sphere
    # rhs = 1

    # Shape-from-shading
    rhs = 1

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


    return d

def alpha():

    return [1, 1, 1]
