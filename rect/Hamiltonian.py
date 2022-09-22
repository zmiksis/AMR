import numpy as np
import itertools

import problem_definition as pd
import subgridclass as subgrid

def LeftRight(grid, node, dim):

    # Get left value
    node_temp = list(node).copy()
    node_temp[dim] = node_temp[dim]-1
    if node_temp[dim] == -1:
        Tn = grid.extrapolateLeft(node,dim)
        Tn = Tn[0]
    else:
        id = grid.findID(node_temp)
        Tn = grid.T[id]

    # Get right value
    node_temp = node.copy()
    node_temp[dim] = node_temp[dim]+1
    if node_temp[dim] == grid.N[dim]+1:
        Tp = grid.extrapolateRight(node,dim)
        Tp = Tp[0]
    else:
        id = grid.findID(node_temp)
        Tp = grid.T[id]

    return [Tn,Tp]

###############################################################################

def dT(grid, T, dim):

    dT = (T[1]-T[0])/(2*grid.h[dim])

    return dT

###############################################################################

def WENOrp(grid, node, dim):

    eps = 1e-6

    id = grid.findID(node)
    T = grid.T[id]

    # Get left value
    node_temp = list(node).copy()
    node_temp[dim] = node_temp[dim]-1
    if node_temp[dim] == -1:
        Tn = grid.extrapolateLeft(node,dim)
        Tn1 = Tn[0]
    else:
        id = grid.findID(node_temp)
        Tn1 = grid.T[id]

    # Get right values
    node_temp = node.copy()
    node_temp[dim] = node_temp[dim]+1
    if node_temp[dim] == grid.N[dim]+1:
        Tp = grid.extrapolateRight(node,dim)
        Tp1 = Tp[0]
        Tp2 = Tp[1]
    elif node_temp[dim] == grid.N[dim]:
        Tp = grid.extrapolateRight(node,dim)
        Tp2 = Tp[0]
        id = grid.findID(node_temp)
        Tp1 = grid.T[id]
    else:
        id = grid.findID(node_temp)
        Tp1 = grid.T[id]
        node_temp[dim] = node_temp[dim]+1
        id = grid.findID(node_temp)
        Tp2 = grid.T[id]

    rp = (eps + (Tp2-2*Tp1+T)**2)/(eps + (Tp1-2*T+Tn1)**2)

    return [rp,Tp1,Tp2]

###############################################################################

def WENOrn(grid, node, dim):

    eps = 1e-6

    id = grid.findID(node)
    T = grid.T[id]

    # Get left value
    node_temp = list(node).copy()
    node_temp[dim] = node_temp[dim]-1
    if node_temp[dim] == -1:
        Tn = grid.extrapolateLeft(node,dim)
        Tn1 = Tn[0]
        Tn2 = Tn[1]
    elif node_temp[dim] == 0:
        id = grid.findID(node_temp)
        Tn1 = grid.T[id]
        Tn = grid.extrapolateLeft(node,dim)
        Tn2 = Tn[0]
    else:
        id = grid.findID(node_temp)
        Tn1 = grid.T[id]
        node_temp[dim] = node_temp[dim]-1
        id = grid.findID(node_temp)
        Tn2 = grid.T[id]

    # Get right values
    node_temp = node.copy()
    node_temp[dim] = node_temp[dim]+1
    if node_temp[dim] == grid.N[dim]+1:
        Tp = grid.extrapolateRight(node,dim)
        Tp1 = Tp[0]
    else:
        id = grid.findID(node_temp)
        Tp1 = grid.T[id]

    rn = (eps + (T-2*Tn1+Tn2)**2)/(eps + (Tp1-2*T+Tn1)**2)

    return [rn, Tn1, Tn2]

###############################################################################

def LaxFriedrichs(grid, node, T, WENO):

    H = 0
    dxT = []
    for i in range(grid.dim):
        if WENO:
            [rp,Tp,Tp2] = WENOrp(grid,node,i)
            wp = 1/(1+2*rp**2)
            [rn,Tn,Tn2] = WENOrn(grid,node,i)
            wn = 1/(1+2*rn**2)

            dxp = (1-wp)*((Tp-Tn)/(2*grid.h[i])) + wp*((-Tp2+4*Tp-3*T)/(2*grid.h[i]))
            dxn = (1-wn)*((Tp-Tn)/(2*grid.h[i])) + wn*((3*T-4*Tn+Tn2)/(2*grid.h[i]))

            dxT.append((dxp+dxn)/2)
        else:
            [Tn,Tp] = LeftRight(grid,node,i)
            dxT.append(dT(grid,[Tn,Tp],i))
        H = H + pd.alpha()[i]*(Tp-2*T+Tn)/(2*grid.h[i])
    H = pd.Hamiltonian(dxT) - H

    return H
