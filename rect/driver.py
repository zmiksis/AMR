import math
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp
import itertools
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
import cv2

import subgridclass as subgrid
import problem_definition as pd
import Hamiltonian
import mesh_refinement as mr

if __name__ == '__main__':

    # Turn AMR on/off
    AMR = False
    # Use high order sweeping
    highOrder = True
    # Set first order convergence tolerance
    conv_tol = 1e-2
    # Set high order convergence tolerance
    HO_conv_tol = 1e-3

    # Initialize domain and list of grids
    lim = pd.domain()
    N = pd.grid_size()

    # Import image
    sfs = cv2.imread('vase-I.png')
    # sfs = cv2.imread('illumination-low.png')
    # sfs = cv2.imread('vase-oblique-I.png')
    sfs = cv2.resize(sfs,(N[0]+1,N[1]+1))
    # Get normalized B/W values
    R = sfs[:,:,2]
    G = sfs[:,:,1]
    B = sfs[:,:,0]
    I = 0.3*R + 0.59*G + 0.11*B
    I /= 255
    # Rotate so (0,0) is bottom-left of image
    I = list(zip(*I[::-1]))
    # Reset values: 1 - peak, 0 - valley
    I = np.subtract(1,I)

    # Setup initial grid, G[0][0]
    G = [[]]
    grid = subgrid.subgrid(lim,N,I)
    # Compute image radiance
    # I = np.zeros((N[0]+1,N[1]+1))
    # for i in range(grid.N[0]+1):
    #     for j in range(grid.N[1]+1):
    #         node = [i,j]
    #         [Txn,Txp] = Hamiltonian.LeftRightI(grid,node,0)
    #         p = (Txp-Txn)/(2*grid.h[0])
    #         [Txn,Txp] = Hamiltonian.LeftRightI(grid,node,1)
    #         q = (Txp-Txn)/(2*grid.h[1])
    # #         N = np.divide(np.array([p,q,1]),np.sqrt(p**2 + q**2 + 1))
    # #         S = np.array([0,0,1])
    # #         NS = np.dot(N,S)
    # #         rho = 1
    # #         A = 1 # Works well for surface SFS...why?
    # #         I[i][j] = A*rho*NS
    #         if p == 0 and q == 0:
    #             I[i][j] = 0
    #         else:
    #             I[i][j] = 1/np.sqrt(p**2 + q**2 + 1)
    grid.I = np.array(I)  #.copy()
    G[0].append(grid)

    # Setup exact solution
    index = []
    exact_sol = np.ones(len(G[0][0].T))
    for i in range(G[0][0].dim):
        index.append(np.array(range(G[0][0].N[i]+1)))
    index = np.array(index)
    for node in itertools.product(*index):
        x = []
        for i in range(G[0][0].dim):
            x.append(G[0][0].lim[i][0]+node[i]*G[0][0].h[i])
        x = np.array(x)
        id = 0
        for d in range(G[0][0].dim-1):
            id = id + node[d]*(G[0][0].N[d]+1)
        id = id + node[G[0][0].dim-1]
        exact_sol[id] = pd.exact(x)

    # Setup initial values
    iterate = 0
    T_old = G[0][0].T.copy()
    # Start timer
    t0 = time.time()

    # Implement sweeping here
    G[0][0].sweep(1)
    iterate = iterate + 1
    while max(abs(G[0][0].T-T_old)) > conv_tol:
        T_old = G[0][0].T.copy()
        G[0][0].sweep(1)
        iterate = iterate + 1
        print('G-S convergence error:',max(abs(G[0][0].T-T_old)))

    G[0][0].sweep(2)
    iterate = iterate + 1
    while max(abs(G[0][0].T-T_old)) > conv_tol:
        T_old = G[0][0].T.copy()
        G[0][0].sweep(2)
        iterate = iterate + 1
        print('Pass 2 G-S convergence error:',max(abs(G[0][0].T-T_old)))

    if highOrder:
        T_old = G[0][0].T.copy()
        G[0][0].sweep(1,True)
        iterate = iterate + 1
        while max(abs(G[0][0].T-T_old)) > HO_conv_tol:
            T_old = G[0][0].T.copy()
            G[0][0].sweep(1,True)
            iterate = iterate + 1
            print('WENO G-S convergence error:',max(abs(G[0][0].T-T_old)))

        T_old = G[0][0].T.copy()
        G[0][0].sweep(2,True)
        iterate = iterate + 1
        while max(abs(G[0][0].T-T_old)) > HO_conv_tol:
            T_old = G[0][0].T.copy()
            G[0][0].sweep(2,True)
            iterate = iterate + 1
            print('Pass 2 WENO G-S convergence error:',max(abs(G[0][0].T-T_old)))

    if AMR:
        # Only refine once (L^8 error not changing)
        # ref_lvl = 0
        for ref_lvl in range(2):
            # Create new list of subgrids
            grid = subgrid.subgrid(G[ref_lvl][0].lim,np.multiply(2,G[ref_lvl][0].N))
            G.append([grid])
            ref_list = []
            # Identify points that need refinement
            if ref_lvl == 0:
                [G,ref_list] = mr.refinement_list(G,ref_lvl,0,ref_list)
                G = mr.generate_subgrids(G,ref_lvl,0)
            else:
                for ref_grid in range(1,len(G[ref_lvl])):
                    [G,ref_list] = mr.refinement_list(G,ref_lvl,ref_grid,ref_list)
                    G = mr.generate_subgrids(G,ref_lvl,ref_grid)
            for p in ref_list:
                node = G[ref_lvl][0].findNode(p)
                G[ref_lvl][0].ref_flags[node[0]][node[1]] = 1
            # Create new subgrids
            # G = mr.generate_subgrids(G,ref_lvl,0)

            fig4, ax4 = plt.subplots()
            ax4.set_aspect('equal')
            ax4.set_title('Mesh')
            xy = []
            index = []
            for i in range(G[0][0].dim):
                index.append(np.array(range(G[0][0].N[i]+1)))
            index = np.array(index, dtype=object)
            for node in itertools.product(*index):
                nodex = []
                for i in range(G[0][0].dim):
                    nodex.append(G[0][0].lim[i][0]+node[i]*G[0][0].h[i])
                nodex = np.array(nodex)
                xy.append(nodex)
            x = np.array([item[0] for item in xy])
            y = np.array([item[1] for item in xy])
            for i in range(len(y)):
                ax4.plot([min(x),max(x)],[y[i],y[i]],'r-',linewidth=0.5)
            for i in range(len(x)):
                ax4.plot([x[i],x[i]],[min(y),max(y)],'r-',linewidth=0.5)
            # Plot new grids
            for l in range(1,len(G)):
                for g in range(1,len(G[l])):
                    xy = []
                    index = []
                    for i in range(G[l][g].dim):
                        index.append(np.array(range(G[l][g].N[i]+1)))
                    index = np.array(index, dtype=object)
                    for node in itertools.product(*index):
                        nodex = []
                        for i in range(G[l][g].dim):
                            nodex.append(G[l][g].lim[i][0]+node[i]*G[l][g].h[i])
                        nodex = np.array(nodex)
                        xy.append(nodex)
                    x = np.array([item[0] for item in xy])
                    y = np.array([item[1] for item in xy])
                    for i in range(len(y)):
                        ax4.plot([min(x),max(x)],[y[i],y[i]],'r-',linewidth=0.5)
                    for i in range(len(x)):
                        ax4.plot([x[i],x[i]],[min(y),max(y)],'r-',linewidth=0.5)
            # for p in ref_list:
            #     ax4.plot(p[0],p[1],'bo')
            # plt.savefig('mesh.png')

            # Correction sweeping
            T_old = G[0][0].T.copy()
            for lvl in range(ref_lvl+1,0,-1):
                for g in range(1,len(G[ref_lvl+1])):
                    G[lvl][g].sweep(1)
                    G = mr.send_values(G,lvl,g)
            G[0][0].sweep(1)
            iterate = iterate + 1
            while max(abs(G[0][0].T-T_old)) > 1e-13:
                T_old = G[0][0].T.copy()
                for lvl in range(ref_lvl+1,0,-1):
                    for g in range(1,len(G[lvl])):
                        G = mr.get_values(G,lvl,g)
                        G[lvl][g] = mr.interpolate_interior(G[lvl][g],G,lvl-1,0)
                        G[lvl][g].sweep(1)
                        G = mr.send_values(G,lvl,g)
                G[0][0].sweep(1)
                iterate = iterate + 1
                print('Correction G-S convergence error:',max(abs(G[0][0].T-T_old)))

    # Stop timer
    t1 = time.time()

    # total_error = 0
    # total_points = 0
    # max_error = 0
    # for i in range(G[0][0].N[0]+1):
    #     for j in range(G[0][0].N[1]+1):
    #         node = np.array([i,j])
    #         id = G[0][0].findID(node)
    #         x = G[0][0].findX(node)
    #         exactT = pd.exact(x)
    #         d1 = math.dist(x,[math.sqrt(1.5),0])
    #         d2 = math.dist(x,[-1,0])
    #         if d1 > 0.15 and d2 > 0.15 and abs(d1-d2) > 0.3:
    #             total_error += abs(G[0][0].T[id] - exactT)
    #             total_points += 1
    #             max_error = max(max_error,abs(G[0][0].T[id] - exactT))
    # avg_error = total_error/total_points

    # Write results to file
    if G[0][0].dim == 1:
        gridstring = ""
        gridnodestring = "Initial grid size:       {} nodes".format(G[0][0].N[0])
        fingridstring = "\nFinal grid size:         {} nodes".format(G[0][0].N[0])
    if G[0][0].dim == 2:
        node_total = (G[0][0].N[0]+1)*(G[0][0].N[1]+1)
        for l in range(1,len(G)):
            for i in range(1,len(G[l])):
                node_total += (G[l][i].N[0]+1)*(G[l][i].N[1]+1)
                node_total -= (G[l][i].N[0]/2+1)*(G[l][i].N[1]/2+1)
        gridstring = "Initial grid dimensions: {}x{}".format(G[0][0].N[0],G[0][0].N[1])
        gridnodestring = "\nInitial grid size:       {} nodes".format((G[0][0].N[0]+1)*(G[0][0].N[1]+1))
        fingridstring = "\nFinal grid size:         {} nodes".format(int(node_total))
    if G[0][0].dim == 3:
        gridstring = "Initial grid dimensions: {}x{}x{}".format(G[0][0].N[0],G[0][0].N[1],G[0][0].N[2])
        gridnodestring = "\nInitial grid size:       {} nodes".format(G[0][0].N[0]*G[0][0].N[1]*G[0][0].N[2])
        fingridstring = "\nFinal grid size:         {} nodes".format(G[0][0].N[0]*G[0][0].N[1]*G[0][0].N[2])
    gammastring = "\nRK gamma value:          {}".format(pd.gamma())
    itstring = "\nIterations:              {}".format(iterate)
    L1string = "\nL^1 error:               {:.5e}".format(np.linalg.norm(G[0][0].T-exact_sol,ord=1)/len(G[0][0].T))
    Linfstring = "\nL^inf error:             {:.5e}".format(max(abs(G[0][0].T-exact_sol)))
    # L1string = "\nL^1 error:               {:.5e}".format(avg_error)
    # Linfstring = "\nL^inf error:             {:.5e}".format(max_error)
    CPUstring = "\nCPU sec:                 {:.5f}\n".format(t1-t0)
    blankLine = "\n"

    f = open("results.txt", "a")
    f.writelines([gridstring,gridnodestring,fingridstring,gammastring,itstring,
                    L1string,Linfstring,CPUstring,blankLine])
    f.close()

    # Write solution to binary file
    solByteArray = bytearray(G[0][0].T)
    sol = open("sol.bin", "wb")
    sol.write(solByteArray)

    # Generate list of grid nodes for plotting
    xy = []
    index = []
    for i in range(G[0][0].dim):
        index.append(np.array(range(G[0][0].N[i]+1)))
    index = np.array(index)
    for node in itertools.product(*index):
        nodex = []
        for i in range(G[0][0].dim):
            nodex.append(G[0][0].lim[i][0]+node[i]*G[0][0].h[i])
        nodex = np.array(nodex)
        xy.append(nodex)

    # Plot 1D solution
    if G[0][0].dim == 1:
        x = np.array([item[0] for item in xy])
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        ax1.plot(x,G[0][0].T)
        ax1.set_title('Plot of numerical solution')
        plt.savefig('sol.png')

    # Plot 2D solution
    if G[0][0].dim == 2:
        x = np.array([item[0] for item in xy])
        y = np.array([item[1] for item in xy])
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        tcf = ax1.tricontourf(x,y,G[0][0].T)
        fig1.colorbar(tcf)
        ax1.set_title('Contour plot of numerical solution')
        plt.savefig('sol.png')

        # Plot error
        xy = []
        index = []
        for i in range(G[0][0].dim):
            index.append(np.array(range(G[0][0].N[i]+1)))
        index = np.array(index)
        for node in itertools.product(*index):
            nodex = []
            for i in range(G[0][0].dim):
                nodex.append(G[0][0].lim[i][0]+node[i]*G[0][0].h[i])
            nodex = np.array(nodex)
            xy.append(nodex)
        x = np.array([item[0] for item in xy])
        y = np.array([item[1] for item in xy])
        fig2, ax2 = plt.subplots()
        ax2.set_aspect('equal')
        # tcf = ax2.tricontourf(x,y,abs(G[0][0].T-exact_sol), levels=np.linspace(0,0.15,11))
        tcf = ax2.tricontourf(x,y,abs(G[0][0].T-exact_sol))
        fig2.colorbar(tcf)
        ax2.set_title('Contour plot of numerical error')
        plt.savefig('error.png')

        # Compute and plot WENO smoothness indicators
        # r = []
        # for node in itertools.product(*index):
        #     rpx = Hamiltonian.WENOrp(G[0][0],np.array(node),0)
        #     rnx = Hamiltonian.WENOrn(G[0][0],np.array(node),0)
        #     rpy = Hamiltonian.WENOrp(G[0][0],np.array(node),1)
        #     rny = Hamiltonian.WENOrn(G[0][0],np.array(node),1)
        #     r.append(min(rpx,rnx,rpy,rny))
        #     # r.append(max(rpx,rnx,rpy,rny))
        # fig3, ax3 = plt.subplots()
        # ax3.set_aspect('equal')
        # tcf = ax3.tricontourf(x,y,r, levels = np.linspace(0,0.5,11))
        # # tcf = ax3.tricontourf(x,y,r, levels = np.linspace(0,1.5,11))
        # fig3.colorbar(tcf)
        # ax3.set_title('Contour plot of WENO indicator')
        # plt.savefig('indicator.png')

        # fig4, ax4 = plt.subplots()
        # ax4.set_aspect('equal')
        # ax4.set_title('Mesh')
        # xy = []
        # index = []
        # for i in range(G[0][0].dim):
        #     index.append(np.array(range(G[0][0].N[i]+1)))
        # index = np.array(index, dtype=object)
        # for node in itertools.product(*index):
        #     nodex = []
        #     for i in range(G[0][0].dim):
        #         nodex.append(G[0][0].lim[i][0]+node[i]*G[0][0].h[i])
        #     nodex = np.array(nodex)
        #     xy.append(nodex)
        # x = np.array([item[0] for item in xy])
        # y = np.array([item[1] for item in xy])
        # for i in range(len(y)):
        #     ax4.plot([min(x),max(x)],[y[i],y[i]],'r-',linewidth=0.5)
        # for i in range(len(x)):
        #     ax4.plot([x[i],x[i]],[min(y),max(y)],'r-',linewidth=0.5)
        # # Plot new grids
        # for l in range(1,len(G)):
        #     for g in range(1,len(G[l])):
        #         xy = []
        #         index = []
        #         for i in range(G[l][g].dim):
        #             index.append(np.array(range(G[l][g].N[i]+1)))
        #         index = np.array(index, dtype=object)
        #         for node in itertools.product(*index):
        #             nodex = []
        #             for i in range(G[l][g].dim):
        #                 nodex.append(G[l][g].lim[i][0]+node[i]*G[l][g].h[i])
        #             nodex = np.array(nodex)
        #             xy.append(nodex)
        #         x = np.array([item[0] for item in xy])
        #         y = np.array([item[1] for item in xy])
        #         for i in range(len(y)):
        #             ax4.plot([min(x),max(x)],[y[i],y[i]],'r-',linewidth=0.5)
        #         for i in range(len(x)):
        #             ax4.plot([x[i],x[i]],[min(y),max(y)],'r-',linewidth=0.5)
        # # for p in ref_list:
        # #     ax4.plot(p[0],p[1],'bo')
        # plt.savefig('mesh.png')

        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111, projection='3d')
        ax5.set_title('Surface')
        xy = []
        index = []
        for i in range(G[0][0].dim):
            index.append(np.array(range(G[0][0].N[i]+1)))
        index = np.array(index, dtype=object)
        for node in itertools.product(*index):
            nodex = []
            for i in range(G[0][0].dim):
                nodex.append(G[0][0].lim[i][0]+node[i]*G[0][0].h[i])
            nodex = np.array(nodex)
            xy.append(nodex)
        x = np.array([item[0] for item in xy])
        y = np.array([item[1] for item in xy])
        z = G[0][0].T.copy()
        X = np.reshape(x, (G[0][0].N[0]+1,G[0][0].N[1]+1))
        Y = np.reshape(y, (G[0][0].N[0]+1,G[0][0].N[1]+1))
        Z = np.reshape(z, (G[0][0].N[0]+1,G[0][0].N[1]+1))
        ax5.plot_surface(X,Y,Z,cmap='Greys')
        plt.savefig('surface.png')


    # Plot 3D isosurface
    if G[0][0].dim == 3:
        X = np.array([item[0] for item in xy])
        Y = np.array([item[1] for item in xy])
        Z = np.array([item[2] for item in xy])
        tcf = go.Figure(data=go.Isosurface(
            x=X,y=Y,z=Z,value=G[0][0].T,
            isomin=0,isomax=1,
            caps=dict(x_show=False, y_show=False)
        ))
        tcf.update_layout(title_text="Isosurface of numerical solution", title_x=0.5)
        img_bytes = tcf.to_image('png')
        with open("sol.png", "wb") as png:
            png.write(img_bytes)
