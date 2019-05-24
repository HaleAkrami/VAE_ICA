from keras.datasets import mnist
import numpy as np
(X,Y), (x_test, y_test) = mnist.load_data()
X_train = X.reshape((len(X), np.prod(X.shape[1:])))
numclasses=np.amax(Y)-np.amin(Y)
#X_sep=np.zeros((numclasses,X_train.shape[0],X_train.shape[1]))
for i in range(numclasses):
    if i==0:
        indx=np.where(Y==0)
        X_sep=np.zeros((numclasses,len(indx[0]),X_train.shape[1]))
        X_sep[i,:,:]=X_train[indx,:]
    else:
        indx=np.where(Y==0)
        maxind=max(len(indx[0]),X_sep.shape[1])
        X_sep=X_sep[:,:,0:maxind]
        X_sep[i,:,:]=X_train[indx[0:maxind],:]