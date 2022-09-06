import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp
import itertools
import plotly.graph_objects as go

import subgridclass as subgrid
import problem_definition as pd
import Hamiltonian

if __name__ == '__main__':

    # Initialize domain and list of grids
    lim = pd.domain()
    N = pd.grid_size()
    G = [[]]

    # Setup initial grid, G[0][0]
    grid = subgrid.subgrid(lim,N)
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
    G[0][0].sweep()
    iterate = iterate + 1
    while max(abs(G[0][0].T-T_old)) > 1e-13:
        T_old = G[0][0].T.copy()
        G[0][0].sweep()
        iterate = iterate + 1
        print('G-S convergence error:',max(abs(G[0][0].T-T_old)))

    # Identify points that need refinement
    # Generate lists of ordered indexes to check
    index = []
    for i in range(G[0][0].dim):
        index.append(np.array(range(G[0][0].N[i]+1)))
    index = np.array(index)
    # Initiate list of nodes to refine
    ref_list = []
    # Check each node
    for idx in itertools.product(*index):
        node = np.array(idx)
        # Find current grid node
        x = G[0][0].findX(node)
        # Get T at current node
        id = G[0][0].findID(node)
        T = G[0][0].T[id]
        # Compute residual
        f = pd.f(x)
        res = abs(Hamiltonian.LaxFriedrichs(G[0][0],node,T) - f)
        # res = abs(T - pd.exact(x))
        rpx = Hamiltonian.WENOrp(G[0][0],np.array(node),0)
        rnx = Hamiltonian.WENOrn(G[0][0],np.array(node),0)
        rpy = Hamiltonian.WENOrp(G[0][0],np.array(node),1)
        rny = Hamiltonian.WENOrn(G[0][0],np.array(node),1)
        r = min(rpx,rnx,rpy,rny)
        # if res > 7e-2:
        if r < 0.5 and res < 1e-2:
            ref_list.append(x)

    # Create new list of subgrids
    G.append([])
    for i in range(len(ref_list)):
        # Generate base grid
        N = np.repeat(8,G[0][0].dim)
        lim = []
        for d in range(G[0][0].dim):
            lim1 = ref_list[i][d]-2*G[0][0].h[d]
            if lim1 < G[0][0].lim[d][0]:
                lim1 = G[0][0].lim[d][0]
            lim2 = ref_list[i][d]+2*G[0][0].h[d]
            if lim2 > G[0][0].lim[d][1]:
                lim2 = G[0][0].lim[d][1]
            lim.append([lim1,lim2])
        grid = subgrid.subgrid(lim,N)

        index = []
        for d in range(grid.dim):
            index.append(np.array(range(grid.N[d]+1)))
        index = np.array(index)
        # Copy computed values from coarser grid
        for idx in itertools.product(*index):
            node = np.array(idx)
            # Find current grid node
            x = grid.findX(node)
            if G[0][0].checkNode(x):
                nodeG = G[0][0].findNode(x)
                idG = G[0][0].findID(nodeG)
                id = grid.findID(node)
                grid.T[id] = G[0][0].T[idG]
                grid.locked.append(node)
        # Interpolate boundary nodes
        for idx in itertools.product(*index):
            node = np.array(idx)
            # Check each dimension
            for d in range(grid.dim):
                # Check if boundary
                if node[d] == 0 or node[d] == grid.N[d]:
                    # Check every other dimension
                    for dd in range(grid.dim):
                        # Check that the node is an interior node
                        if dd != d and not node[dd]%2 == 0:
                            [Tn,Tp] = Hamiltonian.LeftRight(grid,node,d)
                            id = grid.findID(node)
                            grid.T[id] = (Tp+Tn)/2
                            grid.locked.append(node)
        G[1].append(grid)

    plt.show()

    # for g in range(len(G[1])):
    #     # Setup initial values
    #     T_old = G[1][g].T.copy()
    #
    #     # Implement sweeping here
    #     G[1][g].sweep()
    #     iterate = iterate + 1
    #     while max(abs(G[1][g].T-T_old)) > 1e-13:
    #         T_old = G[1][g].T.copy()
    #         G[1][g].sweep()
    #         iterate = iterate + 1
    #         print("G-S convergence error of grid G(1,{}): {}".format(g,max(abs(G[1][g].T-T_old))))
    #
    #     # Send values back to G[0]
    #     index = []
    #     for i in range(G[1][g].dim):
    #         index.append(np.array(range(G[1][g].N[i]+1)))
    #     index = np.array(index)
    #     # Copy computed values from coarser grid
    #     for idx in itertools.product(*index):
    #         node = np.array(idx)
    #         # Find current grid node
    #         x = G[1][g].findX(node)
    #         if G[0][0].checkNode(x):
    #             nodeG = G[0][0].findNode(x)
    #             idG = G[0][0].findID(nodeG)
    #             id = grid.findID(node)
    #             G[0][0].T[idG] = min(grid.T[id], G[0][0].T[idG])

    # Stop timer
    t1 = time.time()

    # Write results to file
    if G[0][0].dim == 1:
        gridstring = ""
        gridnodestring = "Initial grid size:       {} nodes".format(G[0][0].N[0])
        fingridstring = "\nFinal grid size:         {} nodes".format(G[0][0].N[0])
    if G[0][0].dim == 2:
        node_total = 0
        for l in range(len(G)):
            for i in range(len(G[l])):
                node_total = node_total + G[l][i].N[0]*G[l][i].N[1]
        gridstring = "Initial grid dimensions: {}x{}".format(G[0][0].N[0],G[0][0].N[1])
        gridnodestring = "\nInitial grid size:       {} nodes".format(G[0][0].N[0]*G[0][0].N[1])
        fingridstring = "\nFinal grid size:         {} nodes".format(node_total)
    if G[0][0].dim == 3:
        gridstring = "Initial grid dimensions: {}x{}x{}".format(G[0][0].N[0],G[0][0].N[1],G[0][0].N[2])
        gridnodestring = "\nInitial grid size:       {} nodes".format(G[0][0].N[0]*G[0][0].N[1]*G[0][0].N[2])
        fingridstring = "\nFinal grid size:         {} nodes".format(G[0][0].N[0]*G[0][0].N[1]*G[0][0].N[2])
    gammastring = "\nRK gamma value:          {}".format(pd.gamma())
    itstring = "\nIterations:              {}".format(iterate)
    L1string = "\nL^1 error:               {:.5e}".format(np.linalg.norm(G[0][0].T-exact_sol,ord=1)/len(G[0][0].T))
    Linfstring = "\nL^inf error:             {:.5e}".format(max(abs(G[0][0].T-exact_sol)))
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
        # Plot nodes and new grids
        ref_x = np.array([item[0] for item in ref_list])
        ref_y = np.array([item[1] for item in ref_list])
        for l in range(len(G)):
            for g in range(len(G[l])):
                xy = []
                index = []
                for i in range(G[l][g].dim):
                    index.append(np.array(range(G[l][g].N[i]+1)))
                index = np.array(index)
                for node in itertools.product(*index):
                    nodex = []
                    for i in range(G[l][g].dim):
                        nodex.append(G[l][g].lim[i][0]+node[i]*G[l][g].h[i])
                    nodex = np.array(nodex)
                    xy.append(nodex)
                x = np.array([item[0] for item in xy])
                y = np.array([item[1] for item in xy])
                for i in range(len(y)):
                    ax1.plot([min(x),max(x)],[y[i],y[i]],'r-',linewidth=0.5)
                for i in range(len(x)):
                    ax1.plot([x[i],x[i]],[min(y),max(y)],'r-',linewidth=0.5)
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
        tcf = ax2.tricontourf(x,y,abs(G[0][0].T-exact_sol))
        fig2.colorbar(tcf)
        ax2.set_title('Contour plot of numerical error')
        plt.savefig('error.png')

        # Compute and plot WENO smoothness indicators
        r = []
        for node in itertools.product(*index):
            rpx = Hamiltonian.WENOrp(G[0][0],np.array(node),0)
            rnx = Hamiltonian.WENOrn(G[0][0],np.array(node),0)
            rpy = Hamiltonian.WENOrp(G[0][0],np.array(node),1)
            rny = Hamiltonian.WENOrn(G[0][0],np.array(node),1)
            r.append(min(rpx,rnx,rpy,rny))
            # r.append(max(rpx,rnx,rpy,rny))
        fig3, ax3 = plt.subplots()
        ax3.set_aspect('equal')
        tcf = ax3.tricontourf(x,y,r, levels = np.linspace(0,0.5,11))
        # tcf = ax3.tricontourf(x,y,r, levels = np.linspace(0,1.5,11))
        fig3.colorbar(tcf)
        ax3.set_title('Contour plot of WENO indicator')
        plt.savefig('indicator.png')


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