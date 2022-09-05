import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math

import solver

def linear_derivative(T,triang,nodes):

    Q = np.array([triang.x[nodes[0]],triang.y[nodes[0]],T[nodes[0]]])
    R = np.array([triang.x[nodes[1]],triang.y[nodes[1]],T[nodes[1]]])
    S = np.array([triang.x[nodes[2]],triang.y[nodes[2]],T[nodes[2]]])

    QR = R - Q
    QS = S - Q
    n = np.cross(QR,QS)

    Tx = n[0]
    Ty = n[1]

    return [Tx, Ty]

# Refine specific list of elements in mesh
def target_refine(triang, refine_list, T, ax, bx, ay, by):

    xv = []
    yv = []

    # Refine each element, and add new nodes to list
    for i in range(len(refine_list)):
        nodes = triang.triangles[refine_list[i]]
        # Conformal adaptation
        triang_refine = tri.Triangulation(triang.x[nodes], triang.y[nodes])
        refiner = tri.UniformTriRefiner(triang_refine)
        triang_refine = refiner.refine_triangulation(subdiv=1)

        xv.extend(triang_refine.x.copy().tolist())
        yv.extend(triang_refine.y.tolist())

        # Do not connect new boundary nodes to old nodes (keep all triangles acute)
        # Give new triangles correct node numbers
        # for j in range(len(triang_refine.triangles)):
        #     for k in range(3):
        #         if triang_refine.triangles[j][k] in (0,1,2):
        #             triang_refine.triangles[j][k] = nodes[triang_refine.triangles[j][k]]
        #         else:
        #             triang_refine.triangles[j][k] = triang_refine.triangles[j][k] + len(triang.x) - 3
        # # Add new nodes to list
        # triang.x = np.append(triang.x,triang_refine.x[3:6],axis=0).copy()
        # triang.y = np.append(triang.y,triang_refine.y[3:6],axis=0).copy()
        # # Add new triangles to list
        # triang.triangles = np.append(triang.triangles,triang_refine.triangles,axis=0).copy()

        # Steiner/Delaunay adaptation
        # A = [triang.x[nodes[0]],triang.y[nodes[0]]]
        # B = [triang.x[nodes[1]],triang.y[nodes[1]]]
        # C = [triang.x[nodes[2]],triang.y[nodes[2]]]
        # D = 2*(A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
        # Ux = ((A[0]**2 + A[1]**2)*(B[1]-C[1]) + (B[0]**2 + B[1]**2)*(C[1]-A[1]) +
        #         (C[0]**2 + C[1]**2)*(A[1]-B[1]))/D
        # Uy = ((A[0]**2 + A[1]**2)*(C[0]-B[0]) + (B[0]**2 + B[1]**2)*(A[0]-C[0]) +
        #         (C[0]**2 + C[1]**2)*(B[0]-A[0]))/D
        # if ax <= Ux <= bx and ay <= Uy <= by:
        #     xv.append(Ux)
        #     yv.append(Uy)
        # else:
        #     if Ux > bx: Ux = bx
        #     if Ux < ax: Ux = ax
        #     if Uy > by: Uy = by
        #     if Uy < ay: Uy = ay
        #     xv.append(Ux)
        #     yv.append(Uy)

    # Remove refined triangle from list
    # refine_list = sorted(refine_list, reverse=True)
    # triang.triangles = triang.triangles.tolist()
    # for i in range(len(refine_list)):
    #     triang.triangles.pop(refine_list[i])
    # triang.triangles = np.array(triang.triangles)
    # # Add new edges to list by retriangulation
    # triang = tri.Triangulation(triang.x,triang.y,triang.triangles)

    xv = [round(item,4) for item in xv]
    yv = [round(item,4) for item in yv]
    xy = tuple(zip(xv,yv))
    res = [idx for idx, val in enumerate(xy) if val in xy[:idx]]

    xv.extend(triang.x.copy().tolist())
    yv.extend(triang.y.copy().tolist())

    # Remove numerically duplicate nodes from list
    xv = [round(item,4) for item in xv]
    yv = [round(item,4) for item in yv]

    xy = tuple(zip(xv,yv))
    xy = list(set([i for i in xy]))

    # Copy old solution to new grid
    xy_old = tuple(zip(triang.x,triang.y))
    T_new = 10*np.ones(len(xy))
    for i in range(len(xy)):
        if xy[i] in xy_old:
            T_new[i] = T[xy_old.index(xy[i])].copy()
    T = T_new.copy()

    # T_new = 10*np.ones(len(triang.x))
    # for i in range(len(T)):
    #     T_new[i] = T[i].copy()
    # T = T_new.copy()

    # Create new grid
    xv = np.array(xy)[:,0]
    yv = np.array(xy)[:,1]
    triang = tri.Triangulation(xv,yv)

    # Assign new nodes with average of value of surrounding nodes
    for i in range(len(T)):
        if T[i] == 10:
            connect_list = np.where(triang.edges == i)[0]
            vals = []
            for j in range(len(connect_list)):
                node = triang.edges[connect_list[j]].tolist()
                node.remove(i)
                vals.append(T[node[0]])
            T[i] = sum(vals)/len(vals)

    return [triang, T]

if __name__ == '__main__':
    print('Main driver not run')
    pass
