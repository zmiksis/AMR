import math
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp
from scipy.spatial import distance
import itertools
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
import copy

import problem_definition as pd
import subgridclass as subgrid
import Hamiltonian

###############################################################################

def refinement_list(grid):

    # Generate lists of ordered indexes to check
    index = []
    for i in range(grid.dim):
        index.append(np.array(range(grid.N[i]+1)))
    index = np.array(index, dtype=object)
    # Initiate list of nodes to refine
    ref_list = []
    # Check each node
    for idx in itertools.product(*index):
        node = np.array(idx)
        # Find current grid node
        x = grid.findX(node)
        # Get T at current node
        id = grid.findID(node)
        T = grid.T[id]
        # Compute residual
        f = pd.f(x)
        # res = abs(Hamiltonian.LaxFriedrichs(G[0][0],node,T) - f)
        res = abs(T - pd.exact(x))
        # rpx = Hamiltonian.WENOrp(grid,np.array(node),0)
        # rnx = Hamiltonian.WENOrn(grid,np.array(node),0)
        # rpy = Hamiltonian.WENOrp(grid,np.array(node),1)
        # rny = Hamiltonian.WENOrn(grid,np.array(node),1)
        # r = min(rpx,rnx,rpy,rny)
        if res > 7e-2:
        # if r < 0.5 and res < 1e-2:
        # if r < 0.05:
            x = [round(item,4) for item in x]
            ref_list.append(x)
            grid.ref_flags[node[0]][node[1]] = 1
            for i in range(-4,5):
                for j in range(-4,5):
                    if 0 <= node[0]+i <= grid.N[0] and 0 <= node[1]+j <= grid.N[1]:
                        grid.ref_flags[node[0]+i][node[1]+j] = 1
                        x = grid.findX(np.array([node[0]+i,node[1]+j]))
                        ref_list.append(x)


    ref_list = np.array(ref_list)

    return ref_list

###############################################################################

def rect_efficiency(grid, rect_bndry):

    rect = grid.ref_flags[rect_bndry[0][0]:rect_bndry[0][1]+1,rect_bndry[1][0]:rect_bndry[1][1]+1].copy()
    efficiency = np.sum(rect)/rect.size

    return [efficiency,rect]

###############################################################################

def rect_signs(rect):

    sigmaH = np.sum(rect,axis=1)
    sigmaV = np.sum(rect,axis=0)

    return [sigmaH,sigmaV]

###############################################################################

def rect_Laplacian(SigmaH, SigmaV):

    LaplacianH = [0]
    LaplacianV = [0]
    for i in range(1,len(SigmaH)-1):
        L = SigmaH[i-1] - 2*SigmaH[i] + SigmaH[i+1]
        LaplacianH.append(L)
    LaplacianH.append(0)
    for i in range(1,len(SigmaV)-1):
        L = SigmaV[i-1] - 2*SigmaV[i] + SigmaV[i+1]
        LaplacianV.append(L)
    LaplacianV.append(0)

    return [LaplacianH,LaplacianV]

###############################################################################

def interpolate_interior(grid, G, init_level, init_grid):

    # Linearly prolongate to refined region
    for jj in range(0,grid.N[1]+1,2):
        for ii in range(0,grid.N[0]+1):
            node = np.array([ii,jj])
            x = grid.findX(node)
            if G[init_level][init_grid].checkNode(x):
                nodeG = G[init_level][init_grid].findNode(x)
                idG = G[init_level][init_grid].findID(nodeG)
                id = grid.findID(node)
                grid.T[id] = G[init_level][init_grid].T[idG]
            else:
                if not pd.gamma_region(x):
                    id = grid.findID(node)
                    node[1] = node[1]+1
                    x = grid.findX(node)
                    nodeG = G[init_level][init_grid].findNode(x)
                    idG = G[init_level][init_grid].findID(nodeG)
                    T = G[init_level][init_grid].T[idG]
                    [Tn,Tp] = Hamiltonian.LeftRight(G[init_level][init_grid],nodeG,1)
                    grid.T[id] = 3*Tn/8 + 3*T/4 - Tp/8
    for ii in range(0,grid.N[0]+1):
        for jj in range(1,grid.N[1]+1,2):
            node = np.array([ii,jj])
            id = grid.findID(node)
            node_shift = np.array([ii,jj-1])
            id_shift = grid.findID(node_shift)
            Tn = grid.T[id_shift]
            node_shift = np.array([ii,jj+1])
            id_shift = grid.findID(node_shift)
            T = grid.T[id_shift]
            if jj == grid.N[1]-1:
                node_shift = np.array([ii,jj-3])
                id_shift = grid.findID(node_shift)
                Tn2 = grid.T[id_shift]
                grid.T[id] = 3*T/8 + 3*Tn/4 - Tn2/8
            else:
                node_shift = np.array([ii,jj+3])
                id_shift = grid.findID(node_shift)
                Tp = grid.T[id_shift]
                grid.T[id] = 3*Tn/8 + 3*T/4 - Tp/8
    # Lock boundaries
    for bnd in [0,grid.N[0]]:
        for bnd_node in range(grid.N[1]+1):
            node = np.array([bnd,bnd_node])
            id = grid.findID(node)
            grid.locked[id] = 1
    for bnd in [0,grid.N[1]]:
        for bnd_node in range(grid.N[0]+1):
            node = np.array([bnd_node,bnd])
            id = grid.findID(node)
            grid.locked[id] = 1

    return grid

###############################################################################

def generate_subgrids(G, init_level, init_grid):

    # Berger and Rigoutsos (1991) clustering
    rect_lims = []
    lim_ind = [[0,G[init_level][0].N[0]],[0,G[init_level][0].N[1]]]
    rect_lims.append(lim_ind)
    for split_rect_original in rect_lims:
        # Cluster flagged points
        [efficiency,rect] = rect_efficiency(G[init_level][0],split_rect_original)
        while efficiency < 0.75:
            split_rect = copy.deepcopy(split_rect_original)
            split = False
            add_rect = False
            # Compute signatures
            [SigmaH,SigmaV] = rect_signs(rect)
            # Check for holes
            if len(np.where(SigmaH == 0)[0]) > 0 and split == False:
                splitL = np.where(SigmaH == 0)[0][0]
                if splitL == 0:
                    splitR = np.nonzero(SigmaH[splitL:])[0][0]
                    split_rect[0][0] += splitR
                elif len(np.nonzero(SigmaH[splitL:])[0]) > 0:
                    splitR = np.nonzero(SigmaH[splitL:])[0][0] + splitL
                    new_rect = [[split_rect[0][0]+splitR,split_rect[0][1]],split_rect[1].copy()]
                    split_rect[0][1] = split_rect[0][0]+splitL-1
                    add_rect = True
                else:
                    split_rect[0][1] = split_rect[0][0]+splitL-1
                split = True
            if len(np.where(SigmaV == 0)[0]) > 0 and split == False:
                splitL = np.where(SigmaV == 0)[0][0]
                if splitL == 0:
                    splitR = np.nonzero(SigmaV[splitL:])[0][0]
                    split_rect[1][0] += splitR
                elif len(np.nonzero(SigmaV[splitL:])[0]) > 0:
                    splitR = np.nonzero(SigmaV[splitL:])[0][0] + splitL
                    new_rect = [split_rect[0].copy(),[split_rect[1][0]+splitR,split_rect[1][1]]]
                    split_rect[1][1] = split_rect[1][0]+splitL-1
                    add_rect = True
                else:
                    split_rect[1][1] = split_rect[1][0]+splitL-1
                split = True
            # If no holes, look for inflection to split
            if split == False:
                # Compute Laplacian of signatures
                [LaplacianH,LaplacianV] = rect_Laplacian(SigmaH,SigmaV)
                # Find biggest inflection point
                max_diff = 0
                max_dim = 0
                split_ind = 0
                for i in range(1,len(LaplacianH)-2):
                    sign1 = np.sign(LaplacianH[i])
                    if sign1 == 0: sign1 = np.sign(LaplacianH[i-1])
                    sign2 = np.sign(LaplacianH[i+1])
                    if sign2 == 0: sign2 = np.sign(LaplacianH[i+2])
                    if sign1 != sign2:
                        diff = abs(LaplacianH[i]-LaplacianH[i+1])
                        if diff > max_diff:
                            max_diff = diff
                            split_ind = i+1
                for i in range(1,len(LaplacianV)-2):
                    sign1 = np.sign(LaplacianV[i])
                    if sign1 == 0: sign1 = np.sign(LaplacianV[i-1])
                    sign2 = np.sign(LaplacianV[i+1])
                    if sign2 == 0: sign2 = np.sign(LaplacianV[i+2])
                    if sign1 != sign2:
                        diff = abs(LaplacianV[i]-LaplacianV[i+1])
                        if diff > max_diff:
                            max_diff = diff
                            split_ind = i+1
                            max_dim = 1
                # Split if inflection point exists
                if max_diff != 0:
                    if max_dim == 0:
                        new_rect = [[split_rect[0][0]+split_ind,split_rect[0][1]],split_rect[1].copy()]
                        split_rect[0][1] = split_rect[0][0]+split_ind-1
                        split = True
                        add_rect = True
                    else:
                        new_rect = [split_rect[0].copy(),[split_rect[1][0]+split_ind,split_rect[1][1]]]
                        split_rect[1][1] = split_rect[1][0]+split_ind-1
                        split = True
                        add_rect = True

            # Compute efficiency of new rectangle
            [efficiency,rect] = rect_efficiency(G[init_level][0],split_rect)
            # Add new rectangle if big enough
            small = 4
            if split == True:
                if new_rect[0][1]-new_rect[0][0]>=small and new_rect[1][1]-new_rect[1][0]>=small:
                    if split_rect[0][1]-split_rect[0][0]>=small and split_rect[1][1]-split_rect[1][0]>=small:
                        split_rect_original[0] = split_rect[0].copy()
                        split_rect_original[1] = split_rect[1].copy()
                        if add_rect: rect_lims.append(new_rect)
                    else:
                        efficiency = 1
                else:
                    efficiency = 1

    # Reset cluster limits
    cluster_lims = []
    for i in range(len(rect_lims)):
        lim = []
        for d in range(G[init_level][init_grid].dim):
            lim1 = G[init_level][init_grid].lim[d][0]+rect_lims[i][d][0]*G[init_level][init_grid].h[d]
            lim1 = round(lim1,4)
            lim2 = G[init_level][init_grid].lim[d][0]+rect_lims[i][d][1]*G[init_level][init_grid].h[d]
            lim2 = round(lim2,4)
            lim.append([lim1,lim2])
        cluster_lims.append(lim)

    for i in range(len(cluster_lims)):
        # Generate base grid
        lim = copy.deepcopy(cluster_lims[i])
        Nx = round(2*(lim[0][1]-lim[0][0])/G[init_level][init_grid].h[0])
        Ny = round(2*(lim[1][1]-lim[1][0])/G[init_level][init_grid].h[1])
        grid = subgrid.subgrid(lim,[Nx,Ny])

        # # Go over x boundary
        # for bnd in [0,grid.N[0]]:
        #     for bnd_node in range(grid.N[1]+1):
        #         node = np.array([bnd,bnd_node])
        #         x = grid.findX(node)
        #         if G[init_level][init_grid].checkNode(x):
        #             nodeG = G[init_level][init_grid].findNode(x)
        #             idG = G[init_level][init_grid].findID(nodeG)
        #             id = grid.findID(node)
        #             grid.T[id] = G[init_level][init_grid].T[idG]
        #             grid.locked[id] = 1
        #         else:
        #             if not pd.gamma_region(x):
        #                 id = grid.findID(node)
        #                 node[1] = node[1]+1
        #                 x = grid.findX(node)
        #                 nodeG = G[init_level][init_grid].findNode(x)
        #                 idG = G[init_level][init_grid].findID(nodeG)
        #                 T = G[init_level][init_grid].T[idG]
        #                 [Tn,Tp] = Hamiltonian.LeftRight(G[init_level][init_grid],nodeG,1)
        #                 grid.T[id] = 3*Tn/8 + 3*T/4 - Tp/8
        #                 grid.locked[id] = 1
        # # Go over y boundary
        # for bnd in [0,grid.N[1]]:
        #     for bnd_node in range(grid.N[0]+1):
        #         node = np.array([bnd_node,bnd])
        #         x = grid.findX(node)
        #         if G[init_level][init_grid].checkNode(x):
        #             nodeG = G[init_level][init_grid].findNode(x)
        #             idG = G[init_level][init_grid].findID(nodeG)
        #             id = grid.findID(node)
        #             grid.T[id] = G[init_level][init_grid].T[idG]
        #             grid.locked[id] = 1
        #         else:
        #             if not pd.gamma_region(x):
        #                 id = grid.findID(node)
        #                 node[0] = node[0]+1
        #                 x = grid.findX(node)
        #                 nodeG = G[init_level][init_grid].findNode(x)
        #                 idG = G[init_level][init_grid].findID(nodeG)
        #                 T = G[init_level][init_grid].T[idG]
        #                 [Tn,Tp] = Hamiltonian.LeftRight(G[init_level][init_grid],nodeG,0)
        #                 grid.T[id] = 3*Tn/8 + 3*T/4 - Tp/8
        #                 grid.locked[id] = 1

        grid = interpolate_interior(grid,G,init_level,init_grid)

        G[init_level+1].append(grid)

    return G

###############################################################################

def generate_lists(G):

    node_list = []
    fine_grid_list = []
    refinedT = []
    X_list = []
    for i in range(G[0][0].N[0]+1):
        for j in range(G[0][0].N[1]+1):
            node = np.array([i,j])
            id = G[0][0].findID(node)
            node_list.append(node)
            fine_grid_list.append([0,0])
            refinedT.append(G[0][0].T[id])
            X_list.append(G[0][0].findX(node))

    ref_lvl = 0

    for ref_grid in range(len(G[ref_lvl+1])):
        for i in range(G[ref_lvl+1][ref_grid].N[0]+1):
            for j in range(G[ref_lvl+1][ref_grid].N[1]+1):
                node = np.array([i,j])
                x = G[ref_lvl+1][ref_grid].findX(node)
                id = np.all(x == X_list,axis=1)
                id = np.where(id == True)[0]
                if id.size > 0:
                    id = id[0]
                    fine_grid_list[id] = [ref_lvl+1,ref_grid]
                    node_list[id] = np.array(node)
                else:
                    id = G[ref_lvl+1][ref_grid].findID(node)
                    node_list.append(node)
                    fine_grid_list.append([ref_lvl+1,ref_grid])
                    refinedT.append(G[ref_lvl+1][ref_grid].T[id])
                    X_list.append(G[ref_lvl+1][ref_grid].findX(node))

    return node_list, fine_grid_list, refinedT, X_list

###############################################################################

def sort_values_l2(X_list, ref_lim):

    ref_pt.append([ref_lim[0,0],ref_lim[1,0]])
    ref_pt.append([ref_lim[0,1],ref_lim[1,0]])
    ref_pt.append([ref_lim[0,0],ref_lim[1,1]])
    ref_pt.append([ref_lim[0,1],ref_lim[1,1]])

    list = []
    for pt in range(len(ref_pt)):
        dist = np.array([])
        for i in range(len(X_list)):
            dist = np.append(dist,math.dist(X_list[i],ref_pt[pt]))
        S = np.argsort(dist,kind='heapsort')
        list.append(S)

    for i in range(len(list)):
        list[i] = list[i].tolist()

    return list

###############################################################################

def send_values(G, level, grid):

    # Send values back to G[0]
    index = []
    for i in range(G[level][grid].dim):
        index.append(np.array(range(G[level][grid].N[i]+1)))
    index = np.array(index, dtype=object)
    # Copy computed values from coarser grid
    for idx in itertools.product(*index):
        node = np.array(idx)
        # Find current grid node
        x = G[level][grid].findX(node)
        if G[0][0].checkNode(x):
            nodeG = G[0][0].findNode(x)
            idG = G[0][0].findID(nodeG)
            id = G[level][grid].findID(node)
            G[0][0].T[idG] = min(G[level][grid].T[id], G[0][0].T[idG])

    return G

###############################################################################

def get_values(G, level, grid):

    # Get values from G[0]
    index = []
    for i in range(G[level][grid].dim):
        index.append(np.array(range(G[level][grid].N[i]+1)))
    index = np.array(index, dtype=object)
    # Copy computed values from coarser grid
    for idx in itertools.product(*index):
        node = np.array(idx)
        # Find current grid node
        x = G[level][grid].findX(node)
        if G[0][0].checkNode(x):
            nodeG = G[0][0].findNode(x)
            idG = G[0][0].findID(nodeG)
            id = G[level][grid].findID(node)
            G[level][grid].T[id] = G[0][0].T[idG]

    return G

###############################################################################

def refine_sweep(G, level, iterate):

    for g in range(len(G[level])):
        # Setup initial values
        T_old = G[level][g].T.copy()

        # Implement sweeping here
        G[level][g].sweep()
        iterate = iterate + 1
        while max(abs(G[level][g].T-T_old)) > 1e-13:
            T_old = G[level][g].T.copy()
            G[level][g].sweep()
            iterate = iterate + 1
            print("G-S convergence error of grid G({},{}): {}".format(level,g,max(abs(G[level][g].T-T_old))))

        # Send values back to G[0]
        G = send_values(G,level,g)

    return [G, iterate]
