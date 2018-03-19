from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np

from svm import getCharLevelLSTMData

print "Getting Data..."
X_train, y_train, X_dev, y_dev, X_test, y_test, maxLength, embeddingLength = getCharLevelLSTMData()


print "Building Model..."
data_dim = embeddingLength
timesteps = maxLength
num_classes = 2


model = Sequential()
model.add(Conv1D(32, kernel_size=(5,), strides=(1,),
                 activation='relu',
                 input_shape=(timesteps, data_dim)))
model.add(MaxPooling1D(pool_size=(2,), strides=(2,)))
model.add(Conv1D(64, (5,), activation='relu'))
model.add(MaxPooling1D(pool_size=(2,)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=64, epochs=100,
          validation_data=(X_dev, y_dev))
