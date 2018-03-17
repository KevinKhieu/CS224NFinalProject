from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import numpy as np

from svm import getMLPData

print "Getting Data..."
X_train, y_train, X_dev, y_dev, X_test, y_test, maxLength, embeddingLength = getMLPData()

print "Building Model..."
data_dim = embeddingLength
timesteps = maxLength
num_classes = 2


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(data_dim,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=64, epochs=100,
          validation_data=(X_dev, y_dev))

print 80 * "="
print "TESTING"
print 80 * "-"
model.evaluate(X_test, y_test, batch_size=64)