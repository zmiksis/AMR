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
        # rpx = Hamiltonian.WENOrp(G[0][0],np.array(node),0)
        # rnx = Hamiltonian.WENOrn(G[0][0],np.array(node),0)
        # rpy = Hamiltonian.WENOrp(G[0][0],np.array(node),1)
        # rny = Hamiltonian.WENOrn(G[0][0],np.array(node),1)
        # r = min(rpx,rnx,rpy,rny)
        if res > 7e-2:
        # if r < 0.5 and res < 1e-2:
        # if r < 0.5:
            x = [round(item,4) for item in x]
            ref_list.append(x)
            grid.ref_flags[node[0]][node[1]] = 1
            for i in range(-3,4):
                for j in range(-3,4):
                    if 0 <= node[0]+i <= grid.N[0] and 0 <= node[1]+j <= grid.N[1]:
                        grid.ref_flags[node[0]+i][node[1]+j] = 1


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

def cluster_points(ref_list, min_dist):

    # Cluster refinement list
    wcss = []
    for number_of_clusters in range(1, 21):
        kmeans = KMeans(n_clusters = number_of_clusters, random_state = 42)
        kmeans.fit(ref_list)
        wcss.append(kmeans.inertia_)
    # Identify elbow
    wcss_curve_max = 0
    clusters = 0
    for i in range(1,len(wcss)-1):
        wcss_curve = wcss[i-1]-2*wcss[i]+wcss[i+1]
        if wcss_curve > wcss_curve_max:
            wcss_curve_max = wcss_curve
            clusters = i+1
    kmeans = KMeans(n_clusters = clusters, random_state = 42)
    kmeans.fit(ref_list)
    # Find maximum distance ratio
    dist = kmeans.transform(ref_list)**2
    max_dist_ratio = 0
    for label in np.unique(kmeans.labels_):
        max_dist = max(dist[kmeans.labels_==label].sum(axis=1))
        mean_dist = np.mean(dist[kmeans.labels_==label].sum(axis=1))
        max_dist_ratio = max(max_dist_ratio,max_dist/mean_dist)
    # Add clusters until maximum ratio is small enough
    # while max_dist_ratio > 1.6:
    while max_dist_ratio > 1.2:
        clusters = clusters + 1
        kmeans = KMeans(n_clusters = clusters, random_state = 42)
        kmeans.fit(ref_list)
        dist = kmeans.transform(ref_list)**2
        max_dist_ratio = 0
        for label in np.unique(kmeans.labels_):
            max_dist = max(dist[kmeans.labels_==label].sum(axis=1))
            mean_dist = np.mean(dist[kmeans.labels_==label].sum(axis=1))
            max_dist_ratio = max(max_dist_ratio,max_dist/mean_dist)

    # sns.scatterplot(x=ref_list[:,0],y=ref_list[:,1],hue=kmeans.labels_)
    # plt.show()

    cluster_list = []
    for i in range(clusters):
        cluster_list.append(ref_list[kmeans.labels_==i])

    for i in range(len(cluster_list)):
        for j in range(i+1,len(cluster_list)):
            if min(distance.cdist(cluster_list[i],cluster_list[j]).min(axis=1)) < min_dist:
                print(i,"too close to",j)

    return cluster_list

###############################################################################

def cluster_limits(grid, cluster_list):

    cluster_lims = []
    for i in range(len(cluster_list)):
        lim1 = [cluster_list[i][0][0],cluster_list[i][0][0]]
        lim2 = [cluster_list[i][0][1],cluster_list[i][0][1]]
        for j in range(1,len(cluster_list[i])):
            lim1 = [min(lim1[0],cluster_list[i][j][0]),max(lim1[1],cluster_list[i][j][0])]
            lim2 = [min(lim2[0],cluster_list[i][j][1]),max(lim2[1],cluster_list[i][j][1])]
        lim1 = [lim1[0]-3*grid.h[0],lim1[1]+3*grid.h[0]]
        lim1 = [round(item,8) for item in lim1]
        lim2 = [lim2[0]-3*grid.h[1],lim2[1]+3*grid.h[1]]
        lim2 = [round(item,8) for item in lim2]
        cluster_lims.append([lim1,lim2])

    return cluster_lims

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
            # Compute signatures
            [SigmaH,SigmaV] = rect_signs(rect)
            # Check for holes
            if len(np.where(SigmaH == 0)[0]) > 0 and split == False:
                splitL = np.where(SigmaH == 0)[0][0]
                if len(np.nonzero(SigmaH[splitL:])[0]) > 0:
                    splitR = np.nonzero(SigmaH[splitL:])[0][0] + splitL
                    new_rect = [[split_rect[0][0]+splitR,split_rect[0][1]],split_rect[1].copy()]
                if splitL > 0:
                    split_rect[0][1] = split_rect[0][0]+splitL-1
                else:
                    split_rect[0][1] = split_rect[0][0]+1
                split = True
            if len(np.where(SigmaV == 0)[0]) > 0 and split == False:
                splitL = np.where(SigmaV == 0)[0][0]
                if len(np.nonzero(SigmaV[splitL:])[0]) > 0:
                    splitR = np.nonzero(SigmaV[splitL:])[0][0] + splitL
                    new_rect = [split_rect[0].copy(),[split_rect[1][0]+splitR,split_rect[1][1]]]
                if splitL > 0:
                    split_rect[1][1] = split_rect[1][0]+splitL-1
                else:
                    split_rect[1][1] = split_rect[1][0]+1
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
                    else:
                        new_rect = [split_rect[0].copy(),[split_rect[1][0]+split_ind,split_rect[1][1]]]
                        split_rect[1][1] = split_rect[1][0]+split_ind-1
                        split = True

            # Compute efficiency of new rectangle
            [efficiency,rect] = rect_efficiency(G[init_level][0],split_rect)
            # Add new rectangle if big enough
            small = 6
            if split == True:
                if new_rect[0][1]-new_rect[0][0]>=small and new_rect[1][1]-new_rect[1][0]>=small:
                    if split_rect[0][1]-split_rect[0][0]>=small and split_rect[1][1]-split_rect[1][0]>=small:
                        rect_lims.append(new_rect)
                        split_rect_original[0] = split_rect[0].copy()
                        split_rect_original[1] = split_rect[1].copy()
                    else:
                        efficiency = 1
                else:
                    efficiency = 1

    # Remove any possible duplicate rectangles
    rect_lims.sort()
    rect_lims = list(rect_lims for rect_lims,_ in itertools.groupby(rect_lims))

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
        # lim = []
        # for d in range(G[init_level][init_grid].dim):
        #     lim1 = cluster_lims[i][d][0]
        #     if lim1 < G[init_level][init_grid].lim[d][0]:
        #         lim1 = G[init_level][init_grid].lim[d][0]
        #     lim2 = cluster_lims[i][d][1]
        #     if lim2 > G[init_level][init_grid].lim[d][1]:
        #         lim2 = G[init_level][init_grid].lim[d][1]
        #     lim.append([lim1,lim2])
        lim = copy.deepcopy(cluster_lims[i])
        Nx = round(2*(lim[0][1]-lim[0][0])/G[init_level][init_grid].h[0])
        Ny = round(2*(lim[1][1]-lim[1][0])/G[init_level][init_grid].h[1])
        grid = subgrid.subgrid(lim,[Nx,Ny])

        # Go over x boundary
        for bnd in [0,grid.N[0]]:
            for bnd_node in range(grid.N[1]+1):
                node = np.array([bnd,bnd_node])
                x = grid.findX(node)
                if G[init_level][init_grid].checkNode(x):
                    nodeG = G[init_level][init_grid].findNode(x)
                    idG = G[init_level][init_grid].findID(nodeG)
                    id = grid.findID(node)
                    grid.T[id] = G[init_level][init_grid].T[idG]
                    grid.locked[id] = 1
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
                        grid.locked[id] = 1
        # Go over y boundary
        for bnd in [0,grid.N[1]]:
            for bnd_node in range(grid.N[0]+1):
                node = np.array([bnd_node,bnd])
                x = grid.findX(node)
                if G[init_level][init_grid].checkNode(x):
                    nodeG = G[init_level][init_grid].findNode(x)
                    idG = G[init_level][init_grid].findID(nodeG)
                    id = grid.findID(node)
                    grid.T[id] = G[init_level][init_grid].T[idG]
                    grid.locked[id] = 1
                else:
                    if not pd.gamma_region(x):
                        id = grid.findID(node)
                        node[0] = node[0]+1
                        x = grid.findX(node)
                        nodeG = G[init_level][init_grid].findNode(x)
                        idG = G[init_level][init_grid].findID(nodeG)
                        T = G[init_level][init_grid].T[idG]
                        [Tn,Tp] = Hamiltonian.LeftRight(G[init_level][init_grid],nodeG,0)
                        grid.T[id] = 3*Tn/8 + 3*T/4 - Tp/8
                        grid.locked[id] = 1

        G[init_level+1].append(grid)

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
        index = []
        for i in range(G[level][g].dim):
            index.append(np.array(range(G[level][g].N[i]+1)))
        index = np.array(index, dtype=object)
        # Copy computed values from coarser grid
        for idx in itertools.product(*index):
            node = np.array(idx)
            # Find current grid node
            x = G[level][g].findX(node)
            if G[0][0].checkNode(x):
                nodeG = G[0][0].findNode(x)
                idG = G[0][0].findID(nodeG)
                id = G[level][g].findID(node)
                G[0][0].T[idG] = min(G[level][g].T[id], G[0][0].T[idG])

    return [G, iterate]
