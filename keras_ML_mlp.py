from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import numpy as np

from svm import getMLPMultiData

print "Getting Data..."
X_train, y_train, X_dev, y_dev, X_test, y_test, maxLength, embeddingLength = getMLPMultiData()

print "Building Model..."
data_dim = embeddingLength
timesteps = maxLength
num_classes = 2

print "Hidden Layers: 3"
print "hidden layer nodes: 32"
print "Learning Rate: 0.01 "
print "Decay: 1e-6"

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(data_dim,)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


model.compile(optimizer=sgd, loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=64, epochs=100,
          validation_data=(X_dev, y_dev))

# model.fit(X_train, y_train,
#           batch_size=64, epochs=100,
#           validation_data=(X_dev, y_dev))