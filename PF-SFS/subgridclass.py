import numpy as np
import itertools
import math

class subgrid:

    def __init__(self, lim, N, I, f):
    # Input:    lim - vector of domain boundaries
    #           N - vector of cell counts
    #           f - focal length

        # Initiate grid constants
        self.dim = len(lim)
        self.lim = lim
        self.N = N
        self.I = np.array(I.copy())
        self.f = f
        self.h = []
        for i in range(self.dim):
            self.h.append((self.lim[i][1]-self.lim[i][0])/N[i])

        # Initiate grid solution values
        Nsize = (self.N[0]+1)*(self.N[1]+1)
        self.locked = np.zeros(Nsize)
        self.ref_flags = np.zeros((N[0]+1,N[1]+1))

        # Initialize sweep orderings
        self.order = np.array(list(itertools.product([-1,1],repeat=self.dim)))

        # Initiate supersolution u0
        # minI = 1
        # for i in range(self.N[0]+1):
        #     for j in range(self.N[1]+1):
        #         if self.I[i,j] > 0 and self.I[i,j] < minI:
        #             minI = self.I[i,j]
        # self.T = -(1/2)*math.log(minI*(self.f**2))*np.ones(Nsize)

        # Initiate supersolution v0
        self.T = np.ones(Nsize)
        for i in range(self.N[0]+1):
            for j in range(self.N[1]+1):
                node = [i,j]
                id = self.findID(node)
                if i == 0 or i == self.N[0] or j == 0 or j == self.N[1]:
                    self.T[id] = 10
                else:
                    if self.I[i,j] > 0:
                        self.T[id] = -((1/2)*math.log(self.I[i,j]) + math.log(self.f))
                    else:
                        self.T[id] = -math.log(self.f)

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

    def sweep(self):

        # Sweep over each direction
        # for dir in range(len(self.order)):
        dir = 0
        # Generate lists of ordered indexes to sweep
        index = []
        for i in range(self.dim):
            if self.order[dir][i] == 1:
                index.append(np.array(range(self.N[i]+1)))
            else:
                index.append(np.array(range(self.N[i],-1,-1)))
        index = np.array(index, dtype=object)

        # Compute on each interior node
        # for idx in itertools.product(*index):
        #     node = np.array(idx)
        for ii in range(1,self.N[0]):

            # print('Sweeping row',ii,'...')

            for jj in range(1,self.N[1]):

                node = np.array([ii,jj])

                # Find current grid node
                x = self.findX(node)
                # Find list id
                id = self.findID(node)
                Tu = self.T[id]

                Q = np.sqrt((self.f**2)/(self.f**2 + x[0]**2 + x[1]**2))
                J = self.I[node[0],node[1]]*(self.f**2)/Q

                D = np.array([[self.f,0],[0,np.sqrt(self.f**2 + x[0]**2 + x[1]**2)]])
                cost = x[1]/np.sqrt(x[0]**2 + x[1]**2)
                sint = -x[0]/np.sqrt(x[0]**2 + x[1]**2)
                R = np.array([[cost,sint],[-sint,cost]])
                Dil = np.matmul(R.transpose(),D)
                Dil = np.matmul(Dil,R)
                Dil *= J

                M = -1e99

                for i in range(50):
                    for j in range(50):

                        # Define function to be minimized
                        # a = ... R^2 variable to minimize over unit ball
                        # Test value
                        xi = -1+i*(2/50)
                        yi = -1+j*(2/50)

                        if xi**2 + yi**2 < 1:
                            a = np.array([xi,yi])

                            fc = -1*np.matmul(Dil,a)
                            lc = -self.I[node[0],node[1]]*(self.f**2)*np.sqrt(1 - a[0]**2 - a[1]**2)
                            s = np.sign(fc)

                            if not np.any(s == 0):
                                nodei = np.array([node[0]+s[0],node[1]])
                                idx = int(self.findID(nodei))
                                Ti = self.T[idx]
                                nodej = np.array([node[0],node[1]+s[1]])
                                idy = int(self.findID(nodej))
                                Tj = self.T[idy]

                                P = np.array([0,0])
                                P[0] = (Tu - Ti)/(-s[0]*self.h[0])
                                P[1] = (Tu - Tj)/(-s[1]*self.h[1])

                                # Maximize M (or minimize -M)
                                Mtemp = -1*np.matmul(fc,P) - lc
                                if Mtemp >= M:
                                    M = Mtemp
                                    dtopt = abs(fc[0])/self.h[0] + abs(fc[1])/self.h[1]
                                    dtopt = 1/dtopt
                                    Qs = np.array([s[0]*Ti/self.h[0],s[1]*Tj/self.h[1]])
                                    c = -dtopt*(np.matmul(fc,Qs) + lc)

                # Solve for new value
                # c = -Tu + dtopt*M

                t = Tu
                gt  = t - dtopt*math.exp(-2*t) + c
                gtp = 1 + 2*dtopt*math.exp(-2*t)
                tk = t - gt/gtp
                while abs(t - tk) > 1e-12:
                    t = tk
                    gt  = t - dtopt*math.exp(-2*t) + c
                    gtp = 1 + 2*dtopt*math.exp(-2*t)
                    tk = t - gt/gtp

                self.T[id] = tk
