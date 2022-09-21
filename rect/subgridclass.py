import numpy as np
import itertools

import problem_definition as pd
import Hamiltonian
import general_functions as gen

class subgrid:

    def __init__(self, lim, N):
    # Input:    lim - vector of domain boundaries
    #           N - vector of cell counts

        # Initiate grid constants
        self.dim = len(lim)
        self.lim = lim
        self.N = N
        self.h = []
        for i in range(self.dim):
            self.h.append((self.lim[i][1]-self.lim[i][0])/N[i])

        # Compute boundary extrapolation constants
        kk = pd.bndry_order()
        self.coel = np.ones(kk+1)
        self.coer = np.ones(kk+1)
        for i in range(1,kk+1):
            for j in range(1,kk+1):
                if j != i:
                    self.coel[i] = (self.coel[i]*(-j))/(i-j)
                    self.coer[i] = (self.coer[i]*(kk+1-j))/(i-j)

        # Initiate grid solution values
        Nsize = 1
        for i in range(self.dim):
            Nsize = Nsize*(self.N[i]+1)
        self.T = 10*np.ones(Nsize)
        self.locked = np.zeros(Nsize)
        self.ref_flags = np.zeros((N[0]+1,N[1]+1))

        # Initialize sweep orderings
        self.order = np.array(list(itertools.product([-1,1],repeat=self.dim)))

        # Generate list of grid nodes
        index = []
        for i in range(self.dim):
            index.append(np.array(range(self.N[i]+1)))
        index = np.array(index, dtype=object)
        for node in itertools.product(*index):
            x = self.findX(node)
            if pd.gamma_region(x):
                id = self.findID(node)
                self.T[id] = pd.exact(x)

###############################################################################

    def findID(self, node):
        # Find index in T for node (i,j,...)
        id = 0
        for d in range(self.dim-1):
            mult = node[d]
            for dd in range(d+1,self.dim):
                mult = mult*(self.N[dd]+1)
            id = id + mult
        id = id + node[self.dim-1]

        return id

###############################################################################

    def findX(self, node):
        # Find coordinates (x,y,...) for node (i,j,...)
        x = []
        for i in range(self.dim):
            x.append(self.lim[i][0]+node[i]*self.h[i])
        x = np.array(x)

        return x

###############################################################################

    def checkNode(self, x):
        # Check if coordinates (x,y,...) are in grid
        check = True
        xn = [round(item,5) for item in x]
        for i in range(self.dim):
            if xn[i] < self.lim[i][0] or xn[i] > self.lim[i][1]:
                check = False
                break
            ni = (xn[i]-self.lim[i][0])/self.h[i]
            ni = round(ni,5)
            if not (ni).is_integer():
                check = False
                break

        return check

###############################################################################

    def findNode(self, x):
        # Find ndoe (i,j,...) for coordinates (x,y,...)
        node = []
        xn = [round(item,5) for item in x]
        for i in range(self.dim):
            ni = (xn[i]-self.lim[i][0])/self.h[i]
            node.append(int(ni))

        return node

###############################################################################

    def extrapolateLeft(self, node, dim):

        Tn = np.zeros(pd.bndry_order())
        for k in range(pd.bndry_order()):
            for j in range(pd.bndry_order()):
                node_temp = list(node).copy()
                node_temp[dim] = j-k
                if node_temp[dim] >= 0:
                    id = self.findID(node_temp)
                    Tn[k] = Tn[k] + self.coel[j+1]*self.T[id]
                else:
                    Tn[k] = Tn[k] + self.coel[j+1]*Tn[k-j-1]

        return Tn

###############################################################################

    def extrapolateRight(self, node, dim):

        Tp = np.zeros(pd.bndry_order())
        for k in range(pd.bndry_order()):
            for j in range(pd.bndry_order()):
                node_temp = list(node).copy()
                node_temp[dim] = self.N[dim]+k-j
                if node_temp[dim] <= self.N[dim]:
                    id = self.findID(node_temp)
                    Tp[k] = Tp[k] + self.coer[pd.bndry_order()-j]*self.T[id]
                else:
                    Tp[k] = Tp[k] + self.coer[pd.bndry_order()-j]*Tp[k-j-1]

        return Tp

###############################################################################

    def sweep(self):

        gamma = pd.gamma()
        # Sweep over each direction
        for dir in range(len(self.order)):
            # Generate lists of ordered indexes to sweep
            index = []
            for i in range(self.dim):
                if self.order[dir][i] == 1:
                    index.append(np.array(range(self.N[i]+1)))
                else:
                    index.append(np.array(range(self.N[i],-1,-1)))
            index = np.array(index, dtype=object)

            # Compute on each node
            for idx in itertools.product(*index):
                node = np.array(idx)
                # Find current grid node
                x = self.findX(node)
                # Find list id
                id = self.findID(node)

                if not pd.gamma_region(x) and self.locked[id] == 0:

                    T = self.T[id]

                    f = pd.f(x)
                    Tbar = f - Hamiltonian.LaxFriedrichs(self,node,T)

                    denom = 0
                    for i in range(self.dim):
                        denom = denom + pd.alpha()[i]/self.h[i]
                    Tbar = Tbar/denom
                    Tbar = gamma*Tbar
                    Tbar = Tbar + T

                    # Find list id
                    id = self.findID(node)
                    self.T[id] = Tbar
