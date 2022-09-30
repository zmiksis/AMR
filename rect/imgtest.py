from matplotlib import pyplot as plt
import numpy as np
import cv2

sfs = cv2.imread('SFS.png')
sfs = cv2.resize(sfs,(41,41))
print(sfs.shape)
R = sfs[:,:,2]
G = sfs[:,:,1]
B = sfs[:,:,0]
I = 0.3*R + 0.59*G + 0.11*B
I /= 255
plt.imshow(sfs)
plt.show()

I = list(zip(*I[::-1]))
plt.imshow(I)
plt.show()
