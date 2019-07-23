from numpy.random import seed
seed(5)
import tensorflow as tf
tf.compat.v1.set_random_seed(6)

import numpy as np
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model, Sequential
from keras import regularizers, optimizers
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import matplotlib.image as image
import logging
import time
from PIL import Image
import cv2
from skimage.measure import compare_ssim as ssim
import os
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system('cls')

###START##################################################################
t = time.time()
Train = np.load('TrainingImageE.npy')
Test = np.load('TestImageE.npy')

max_value = float(Train.max())
x_train = Train.astype('float32') / max_value
ori = Test.astype('float32') / max_value
x_test=ori

# input dimension = 10000
input_dim = x_train.shape[1]
firstlayer = 4800
secondlayer = 3600
encoding_dim = 2400

autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Dense(firstlayer, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(secondlayer, activation='relu'))
autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers
autoencoder.add(Dense(secondlayer, activation='relu'))
autoencoder.add(Dense(firstlayer, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

#Encoder Model
input_img = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

#training & testing
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder_train = autoencoder.fit(x_train, x_train,
                                    epochs=50,
                                    batch_size=256,
                                    shuffle=True,
                                    validation_data=(x_test, x_test),
                                    verbose=0,
                                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

elapsed = time.time() - t

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(50)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Loss Plot NN.jpg')

#printing
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)
np.save('Compressed Image NN.npy',encoded_imgs)

approximation = decoded_imgs.reshape(-1,100, 100)
X_norm = ori.reshape(-1,100, 100)
for i in range(0,X_norm.shape[0]):
    X_norm[i,] = X_norm[i,].T
    approximation[i,] = approximation[i,].T

#######################################
ori2 = X_norm[500,]
recon2 = approximation[500,]

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

print ("Original training image shape: ", x_train.shape, " Original test image shape: ", x_test.shape)
print ("First hidden layer: ", firstlayer," neurons. Second hidden layer: ", secondlayer," neurons. Third hidden layer (compressed layer): ", encoding_dim," neurons")
print ("Reconstructed Image has MSE value of %.2f" % m)
print ("Reconstructed Image has %.2f" % (s*100),"% similarity (SSIM)")
print("Time: %.2f" % (elapsed/3600),"hrs")
##################################
fig4, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(np.flipud(ndimage.rotate((X_norm[0,]),90)),cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(np.flipud(ndimage.rotate((approximation[0,]),90)),cmap='gray')
axarr[0,1].set_title('Dimensionality-Reduced Image')
axarr[0,1].axis('off')
axarr[1,0].imshow(np.flipud(ndimage.rotate((X_norm[1000,]),90)),cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(np.flipud(ndimage.rotate((approximation[1000,]),90)),cmap='gray')
axarr[1,1].set_title('Dimensionality-Reduced Image')
axarr[1,1].axis('off')
axarr[2,0].imshow(np.flipud(ndimage.rotate((X_norm[500,]),90)),cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(np.flipud(ndimage.rotate((approximation[500,]),90)),cmap='gray')
axarr[2,1].set_title('Dimensionality-Reduced Image')
axarr[2,1].axis('off')
plt.savefig('Reproduced Image NN.jpg')
###END####################################################################