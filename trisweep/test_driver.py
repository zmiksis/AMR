import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import scipy as sp

import grid_generator as gridgen
import grid_initializer as gridinit
import solver
import problem_definition as pd
import mesh_refinement as mr

if __name__ == '__main__':

    # grid_list = []
    #
    # ax, bx = (-2, 2)
    # ay, by = (-2, 2)
    #
    # triang = gridgen.gridgen(ax, bx, ay, by)
    #
    # grid_list.append([triang])
    #
    # refiner = tri.UniformTriRefiner(triang)
    # triang = refiner.refine_triangulation(subdiv=1)
    #
    # grid_list.append([triang])
    # grid_list[1].append(triang)
    #
    # print(len(grid_list[1][1].triangles))
    # print(len(grid_list[1][0].triangles))
    #
    # for i in range(len(grid_list[1][1].triangles)):
    #     print('(',grid_list[1][1].x[grid_list[1][1].triangles[i][0]],',',
    #         grid_list[1][1].y[grid_list[1][1].triangles[i][0]],')')


    def arreq_in_list(myarr, list_arrays):
        return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

    list = [np.array(['a','b']),np.array(['c','d'])]
    # print(['a','b'] in list)
    print(arreq_in_list(np.array(['a','b']), list))

    M, N = 5, 8

    x = np.linspace(0,2,M+1)
    y = np.linspace(0,1,N+1)
    # X,Y = np.meshgrid(x,y)

    # Z = X**2 + Y**2

    # positions = np.vstack([Y.ravel(), X.ravel()])
    #
    # is_perimeter = (positions[0] == np.min(y)) | (positions[0] == np.max(y)) | \
    #                (positions[1] == np.min(x)) | (positions[1] == np.max(x))

    # plt.scatter(*positions[::-1], c=is_perimeter)
    # plt.pcolormesh(X,Y,Z,edgecolors='black',alpha=None)
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    for i in range(len(y)):
        ax1.plot([min(x),max(x)],[y[i],y[i]],'k-',linewidth=0.5)
    for i in range(len(x)):
        ax1.plot([x[i],x[i]],[min(y),max(y)],'k-',linewidth=0.5)
    plt.show()
