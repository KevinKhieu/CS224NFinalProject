from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np

from svm import getLSTMMultiData

print "Getting Data..."
X_train, y_train, X_dev, y_dev, X_test, y_test, maxLength, embeddingLength = getLSTMMultiData()

print "Building Model..."
data_dim = embeddingLength
timesteps = maxLength
num_classes = 5

print "Kernal Size: 2,"
print "Stride Size: 1,"
print "Pool Size: 3,"
model = Sequential()
model.add(Conv1D(32, kernel_size=(2,), strides=(1,),
                 activation='relu',
                 input_shape=(timesteps, data_dim)))
model.add(MaxPooling1D(pool_size=(3,), strides=(1,)))
model.add(Conv1D(64, (5,), activation='relu'))
model.add(MaxPooling1D(pool_size=(3,)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# class_weight = {0 : 15,
#     1: 10,
#     2: 2,
#     3: 33,
#     4: 2,
#     5: 13}

model.fit(X_train, y_train,
          batch_size=64, epochs=30,
          validation_data=(X_dev, y_dev))