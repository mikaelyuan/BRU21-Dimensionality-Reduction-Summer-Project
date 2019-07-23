from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.io as sio
import scipy
import scipy.misc
from scipy import ndimage
import matplotlib.image as image
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import cv2
from skimage.measure import compare_ssim as ssim
os.system('cls')

#START####################################################################
inputimage = np.load('ImageA.npy')
max_value = float(inputimage.max())
ori = inputimage.astype('float32') / max_value
print ("Original image shape", ori.shape)
np.save('Original Image PCA.npy',ori)

pca = PCA(.10)
compressed = pca.fit_transform(ori)
NumPCA = len(compressed[0])
recon = pca.inverse_transform(compressed)
np.save('Compressed Image PCA.npy',compressed)

recon = recon.reshape(-1,100, 100)
ori = ori.reshape(-1,100, 100)
for i in range(0,ori.shape[0]):
    ori[i,] = ori[i,].T
    recon[i,] = recon[i,].T

ori2 = ori[500,]
recon2 = recon[500,]

cmap = plt.cm.gray

norm = plt.Normalize(vmin=ori2.min(), vmax=ori2.max())
image = cmap(norm(ori2))
plt.imsave('ori2.png', image)

norm = plt.Normalize(vmin=recon2.min(), vmax=recon2.max())
image = cmap(norm(recon2))
plt.imsave('recon2.png', image)

ori3 = cv2.imread("ori2.png")
recon3 = cv2.imread("recon2.png")

def mse(ori3, recon3):
	err = np.sum((ori3.astype("float") - recon3.astype("float")) ** 2)
	err /= float(ori3.shape[0] * ori3.shape[1])
	return err

m = mse(ori3, recon3)
s = ssim(ori3, recon3, multichannel=True)

print (NumPCA, "Number of PCA will explain", (pca.n_components * 100),"% variation")
print ("Reconstructed Image has MSE value of %.2f" % m)
print ("Reconstructed Image has %.2f" % (s*100),"% similarity (SSIM)")

fig4, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(np.flipud(ndimage.rotate((ori[0,]),90)),cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(np.flipud(ndimage.rotate((recon[0,]),90)),cmap='gray')
axarr[0,1].set_title('Dimensionality-Reduced Image')
axarr[0,1].axis('off')
axarr[1,0].imshow(np.flipud(ndimage.rotate((ori[1000,]),90)),cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(np.flipud(ndimage.rotate((recon[1000,]),90)),cmap='gray')
axarr[1,1].set_title('Dimensionality-Reduced Image')
axarr[1,1].axis('off')
axarr[2,0].imshow(np.flipud(ndimage.rotate((ori[500,]),90)),cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(np.flipud(ndimage.rotate((recon[500,]),90)),cmap='gray')
axarr[2,1].set_title('Dimensionality-Reduced Image')
axarr[2,1].axis('off')
plt.savefig('Reproduced Image PCA.jpg')
#END####################################################################