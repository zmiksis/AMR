 from shadho import Shadho, spaces

 import math
 import numpy as np

 # Need to access passed grid class values, passed node (i,j), and passed (x,y) point

 def obj(params):

     a = np.array([params['a1'],params['a2']])

     Q = np.sqrt(grid.f/(grid.f**2 + x[0]**2 + x[1]**2))
     J = grid.I[node[0],node[1]]*(grid.f**2)/Q

     D = np.array([[grid.f,0],[0,np.sqrt(grid.f**2 + x[0]**2 + x[1]**2)]])
     cost = x[1]/np.sqrt(x[0]**2 + x[1]**2)
     sint = -x[0]/np.sqrt(x[0]**2 + x[1]**2)
     R = np.array([[cost,sint],[-sint,cost]])
     Dil = np.matmul(R.transpose(),D)
     Dil = np.matmul(Dil,R)
     Dil *= J
     fc = -1*np.matmul(Dil,a)
     lc = -grid.I[node[0],node[1]]*(grid.f**2)*np.sqrt(1 - a[0]**2 - a[1]**2)
     s = np.sign(fc)

     nodei = np.array([node[0]+s[0],node[1]])
     idx = int(grid.findID(nodei))
     if idx >= (grid.N[0]+1)*(grid.N[1]+1):
         nodei = np.array([node[0]-s[0],node[1]])
         idx = int(grid.findID(nodei))
     Ti = grid.T[idx]
     nodej = np.array([node[0],node[1]+s[1]])
     idy = int(grid.findID(nodej))
     if idy >= (grid.N[0]+1)*(grid.N[1]+1):
         nodej = np.array([node[0],node[1]-s[1]])
         idy = int(grid.findID(nodej))
     Tj = grid.T[idy]

     P = np.array([0,0])
     P[0] = (Tu - Ti)/(-s[0]*grid.h[0])
     P[1] = (Tu - Tj)/(-s[1]*grid.h[1])

     # Maximize M (or minimize -M)
     M = -1*np.matmul(fc,P) - lc

     return -M

 space = {
     'a1': spaces.uniform(0.0, 1.0),
     'a2': spaces.uniform(0.0, 1.0)
 }


 if __name__ == '__main__':
     opt = Shadho('M_max', obj, space, timeout=60)
     opt.config['global']['manager'] = 'local'
     opt.run()
