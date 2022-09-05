import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import mesh_refinement as mr

def gridgen(ax, bx, ay, by):

    # Acute triangulation of [0,1]x[0,1]
    xv = np.array([0, 1/3, 2/3, 1, 0, 1/3, 2/3, 1, 0, 0, 1, 1,
                    1/5, 1/4, 1/5, 4/5, 3/4, 4/5, 1/2, 1/2])
    yv = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1/3, 2/3, 1/3, 2/3,
                    1/4, 1/2, 3/4, 1/4, 1/2, 3/4, 1/3, 2/3])

    # Create Delaunay triangulation, 26 elements
    # triang = tri.Triangulation(xv, yv)

    # Map gridpoints to [ax,bx]x[ay,by]
    xv = ax + xv*(bx-ax)
    yv = ay + yv*(by-ay)

    # Generate square gridpoints
    # x = np.linspace(ax, bx, Nx)
    # y = np.linspace(ay, by, Ny)
    # xv, yv = np.meshgrid(x, y)
    # xv = np.reshape(xv,(1, Nx*Ny)).flatten()
    # yv = np.reshape(yv,(1, Nx*Ny)).flatten()

    # Create Delaunay triangulation in new space
    triang = tri.Triangulation(xv, yv)

    # Uniformly subdivide triangles into 4, total of 104 elements
    refiner = tri.UniformTriRefiner(triang)
    triang = refiner.refine_triangulation(subdiv=1)

    # Create list of nodal neighbors
    triang.node_neighbors = []
    for i in range(len(xv)):
        temp_list = np.array([])
        neighbors = []
        for j in range(len(np.where(triang.edges == i)[0])):
            temp_neighbor = triang.edges[np.where(triang.edges == i)[0][j]].tolist()
            temp_neighbor.remove(i)
            neighbors.append(temp_neighbor)
        triang.node_neighbors.append(np.array(neighbors).flatten().tolist())

    return triang

if __name__ == '__main__':
    print('Main driver not run')
    pass
