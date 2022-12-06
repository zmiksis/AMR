import math
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import cv2
import open3d as o3d
import os

import subgridclass as subgrid

if __name__ == '__main__':

    # Make surface directory
    path = "surface"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    # Import image
    sfs = cv2.imread('vase001_128.png')
    # sfs = cv2.imread('Mozart001_256.pgm')
    # sfs = cv2.imread('vase256.pgm')
    # sfs = cv2.imread('SFS-I.png')

    # Initialize domain
    N =  np.array([0,0])
    N[0] = sfs.shape[1]-1
    N[1] = sfs.shape[0]-1
    lim = np.array([[-sfs.shape[1]/20,sfs.shape[1]/20],[-sfs.shape[0]/20,sfs.shape[0]/20]])
    #  Focal length
    f = 25

    # Get normalized B/W values
    R = sfs[:,:,2]
    G = sfs[:,:,1]
    B = sfs[:,:,0]
    I = 0.3*R + 0.59*G + 0.11*B
    I /= 255
    I = np.subtract(1,I)
    # I = np.subtract(255,I)
    # Rotate so (0,0) is bottom-left of image
    I = list(zip(*I[::-1]))

    # Setup initial grid
    grid = subgrid.subgrid(lim,N,I,f)

    # Generate list of grid nodes for plotting
    xy = []
    index = []
    for i in range(grid.dim):
        index.append(np.array(range(grid.N[i]+1)))
    index = np.array(index)
    for node in itertools.product(*index):
        nodex = []
        for i in range(grid.dim):
            nodex.append(grid.lim[i][0]+node[i]*grid.h[i])
        nodex = np.array(nodex)
        xy.append(nodex)
    x = np.array([item[0] for item in xy])
    y = np.array([item[1] for item in xy])

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111, projection='3d')
    ax5.set_title('Surface')
    z = grid.T.copy()
    X = np.reshape(x, (grid.N[0]+1,grid.N[1]+1))
    X = X[1:-1,1:-1]
    Y = np.reshape(y, (grid.N[0]+1,grid.N[1]+1))
    Y = Y[1:-1,1:-1]
    Z = np.reshape(z, (grid.N[0]+1,grid.N[1]+1))
    Z = Z[1:-1,1:-1]
    ax5.plot_surface(X,Y,Z,cmap='Greys')
    plt.savefig('surface/super-solution.png')
    fig5.clear()

    surf_array = np.array([x,y,z])
    surf_array = np.transpose(surf_array)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surf_array)
    pcd.estimate_normals()
    # Poisson mesh
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    o3d.io.write_triangle_mesh("surface/p_mesh_super.obj", poisson_mesh)

    # Setup initial values
    iterate = 0
    T_old = grid.T.copy()
    # Start timer
    t0 = time.time()

    # Implement sweeping here
    grid.sweep()
    grid.sweep(True)
    iterate = iterate + 1
    err = np.sum(np.absolute(np.subtract(grid.T,T_old)))/((grid.N[0]+1)*(grid.N[1]+1))
    print('G-S convergence error:',err)
    # print('G-S convergence error:',max(abs(grid.T-T_old)))

    # Plot iteration surface
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111, projection='3d')
    ax5.set_title('Surface')
    z = grid.T.copy()
    X = np.reshape(x, (grid.N[0]+1,grid.N[1]+1))
    X = X[1:-1,1:-1]
    Y = np.reshape(y, (grid.N[0]+1,grid.N[1]+1))
    Y = Y[1:-1,1:-1]
    Z = np.reshape(z, (grid.N[0]+1,grid.N[1]+1))
    Z = Z[1:-1,1:-1]
    ax5.plot_surface(X,Y,Z,cmap='Greys')
    plt.savefig('surface/surface-1.png')
    fig5.clear()

    while iterate < 80:
    # while max(abs(grid.T-T_old)) > conv_tol:
        T_old = grid.T.copy()
        grid.sweep()
        grid.sweep(True)
        iterate = iterate + 1
        err = np.sum(np.absolute(np.subtract(grid.T,T_old)))/((grid.N[0]+1)*(grid.N[1]+1))
        print('G-S convergence error:',err)
        # print('G-S convergence error:',max(abs(grid.T-T_old)))

        # Plot iteration surface
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111, projection='3d')
        ax5.set_title('Surface')
        z = grid.T.copy()
        X = np.reshape(x, (grid.N[0]+1,grid.N[1]+1))
        X = X[1:-1,1:-1]
        Y = np.reshape(y, (grid.N[0]+1,grid.N[1]+1))
        Y = Y[1:-1,1:-1]
        Z = np.reshape(z, (grid.N[0]+1,grid.N[1]+1))
        Z = Z[1:-1,1:-1]
        ax5.plot_surface(X,Y,Z,cmap='Greys')
        fileName = 'surface/surface-' + str(iterate) + '.png'
        plt.savefig(fileName)
        fig5.clear()

    # Stop timer
    t1 = time.time()

    # Write results to file
    node_total = (grid.N[0]+1)*(grid.N[1]+1)
    gridstring = "Initial grid dimensions: {}x{}".format(grid.N[0],grid.N[1])
    gridnodestring = "\nInitial grid size:       {} nodes".format((grid.N[0]+1)*(grid.N[1]+1))
    fingridstring = "\nFinal grid size:         {} nodes".format(int(node_total))
    itstring = "\nIterations:              {}".format(iterate)
    CPUstring = "\nCPU sec:                 {:.5f}\n".format(t1-t0)
    blankLine = "\n"

    f = open("results.txt", "a")
    f.writelines([gridstring,gridnodestring,fingridstring,itstring,
                    CPUstring,blankLine])
    f.close()

    # Write solution to CSV file
    sol = open('sol.csv','w')
    for i in range(grid.N[0]+1):
        x = grid.lim[0][0] + i*grid.h[0]
        for j in range(grid.N[1]+1):
            y = grid.lim[1][0] + j*grid.h[1]
            node = np.array([i,j])
            id = grid.findID(node)
            T = grid.T[id]
            sol.write('%f, %f, %f\n' % (x,y,T))
    sol.close()


    # Generate list of grid nodes for plotting
    xy = []
    index = []
    for i in range(grid.dim):
        index.append(np.array(range(grid.N[i]+1)))
    index = np.array(index)
    for node in itertools.product(*index):
        nodex = []
        for i in range(grid.dim):
            nodex.append(grid.lim[i][0]+node[i]*grid.h[i])
        nodex = np.array(nodex)
        xy.append(nodex)
    x = np.array([item[0] for item in xy])
    y = np.array([item[1] for item in xy])

    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tcf = ax1.tricontourf(x,y,grid.T)
    fig1.colorbar(tcf)
    ax1.set_title('Contour plot of numerical solution')
    plt.savefig('sol.png')

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111, projection='3d')
    ax5.set_title('Surface')
    z = grid.T.copy()
    X = np.reshape(x, (grid.N[0]+1,grid.N[1]+1))
    X = X[1:-1,1:-1]
    Y = np.reshape(y, (grid.N[0]+1,grid.N[1]+1))
    Y = Y[1:-1,1:-1]
    Z = np.reshape(z, (grid.N[0]+1,grid.N[1]+1))
    Z = Z[1:-1,1:-1]
    ax5.plot_surface(X,Y,Z,cmap='Greys')
    plt.savefig('surface/surface.png')
    fig5.clear()

    # delete_index = []
    # for i in range(grid.N[0]+1):
    #     for j in range(grid.N[1]+1):
    #         if(abs(1-grid.I[i][j]) < 1e-2):
    #             idx = i*(grid.N[1]+1) + j
    #             delete_index.append(idx)
    # x = np.delete(x,delete_index)
    # y = np.delete(y,delete_index)
    # z = np.delete(z,delete_index)
    surf_array = np.array([x,y,z])
    surf_array = np.transpose(surf_array)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surf_array)
    pcd.estimate_normals()
    # Poisson mesh
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    o3d.io.write_triangle_mesh("surface/p_mesh_c.obj", poisson_mesh)
    # o3d.io.write_triangle_mesh("surface/p_mesh_c.obj", p_mesh_crop)
