from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from numpy import argmax

X_train, X_test, y_train, y_test = np.load('./binary_image_data.npy',allow_pickle=True)


X_train = X_train.reshape(4465,12288)
X_test = X_test.reshape(497,12288)
X_train = X_train/255.0
X_test = X_test/255.0
y_train = tf.keras.utils.to_categorical(y_train, 5)
y_test = tf.keras.utils.to_categorical(y_test,5)
#print(X_test)

n_input = 12288
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 256
n_hidden4 = 256
n_output = 5

mlp = Sequential()
mlp.add(Dense(units = n_hidden1, activation = 'tanh',
              input_shape=(n_input,), kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_hidden2, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer ='zeros'))
mlp.add(Dense(units = n_hidden3, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer ='zeros'))
mlp.add(Dense(units = n_hidden4, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer ='zeros'))
mlp.add(Dense(units = n_output, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer ='zeros'))

mlp.compile(loss = 'mse', optimizer = 'sgd', metrics = ['accuracy'])
hist = mlp.fit(X_train, y_train, batch_size = 64, epochs = 50 , validation_data = (X_test,y_test), verbose = 2)

res = mlp.evaluate(X_test, y_test, verbose = 0)
print("Accuracy is", res[1]*100)

categories = ['chicken', 'gimbab', 'kimchi', 'mandu', 'ramen']

xhat_idx = np.random.choice(X_test.shape[0], 50)
xhat_value = X_test[xhat_idx]
xhat = X_test[xhat_idx]
yhat = mlp.predict_classes(xhat)



print('True : ' + categories[int(argmax(y_test[xhat_idx[0]]))] + ', Predict : ' + categories[int(yhat[0])])



myfood = xhat_value[0].reshape(64, 64, 3)
plt.imshow(myfood,cmap='gray')
plt.show()


























   