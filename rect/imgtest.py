from matplotlib import pyplot as plt
import numpy as np
import cv2

N = 160
sfs = cv2.imread('vase-I.png')
sfs = cv2.resize(sfs,(N+1,N+1))
print(sfs.shape)
# plt.imshow(sfs)
# plt.axis('off')
# plt.show()
# plt.savefig('reduce.png', bbox_inches='tight')
# plt.savefig('illumination.png', bbox_inches='tight')

R = sfs[:,:,2]
G = sfs[:,:,1]
B = sfs[:,:,0]
Itmp = 0.3*R + 0.59*G + 0.11*B
Itmp /= 255
# Itmp = list(zip(*Itmp[::-1]))
# I = np.zeros((N,N))
# for i in range(N):
#     for j in range(N):
#         p = (Itmp[i+2][j+1] - Itmp[i][j+1])/(2/N)
#         q = (Itmp[i+1][j+2] - Itmp[i+1][j])/(2/N)
#         I[i][j] = 1/np.sqrt(p**2 + q**2 + 1)
        # if Itmp[i+2][j] == 0 and Itmp[i][j+2] == 0 and Itmp[i][j] == 0:
        #     I[i][j] = 0
# Nx = sfs.shape[0]
# Ny = sfs.shape[1]
# I = np.zeros((Nx-2,Ny-2))
# for i in range(Nx-2):
#     for j in range(Ny-2):
#         p = (Itmp[i+2][j+1] - Itmp[i][j+1])/(2/Nx)
#         q = (Itmp[i+1][j+2] - Itmp[i+1][j])/(2/Ny)
#         I[i][j] = (1-p)/np.sqrt(p**2 + q**2 + 1)
# I = np.subtract(1,I)
I = np.array(Itmp)
I = np.subtract(1,I)
print(I)
plt.imshow(I)
plt.axis('off')
plt.set_cmap('Greys')
plt.show()
# plt.savefig('illumination.png', bbox_inches='tight')
