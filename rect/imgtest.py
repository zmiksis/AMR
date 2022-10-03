from matplotlib import pyplot as plt
import numpy as np
import cv2

N = 161
sfs = cv2.imread('vase.png')
sfs = cv2.resize(sfs,(N+2,N+2))
print(sfs.shape)
plt.imshow(sfs)
plt.savefig('reduce.png')

R = sfs[:,:,2]
G = sfs[:,:,1]
B = sfs[:,:,0]
Itmp = 0.3*R + 0.59*G + 0.11*B
Itmp /= 255
# Itmp = list(zip(*Itmp[::-1]))
I = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        p = (Itmp[i+2][j] - Itmp[i][j])/(2/N)
        q = (Itmp[i][j+2] - Itmp[i][j])/(2/N)
        I[i][j] = 1/np.sqrt(p**2 + q**2 + 1)
# I = np.divide(I,np.max(I))
# I = np.subtract(1,I)
# Imin = np.min(Itmp)
# Imax = np.max(Itmp)
# I = np.subtract(Itmp,Imin)
# I = np.divide(I,(Imax-Imin))
print(I)
plt.imshow(I)
plt.savefig('illumination.png',cmap="Greys")
