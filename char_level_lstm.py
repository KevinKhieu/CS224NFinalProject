# import sys
# import numpy as np
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.callbacks import ModelCheckpoint
# from keras.layers import Dense, Dropout, LSTM
# from keras.layers.embeddings import Embedding


# from svm import 


from keras.models import Sequential
from keras.layers import *
import numpy as np

from svm import getCharLevelLSTMData

print "Getting Data..."
X_train, y_train, X_dev, y_dev, X_test, y_test, maxLength, embeddingLength = getCharLevelLSTMData()

print "Building Model..."

data_dim = embeddingLength
timesteps = maxLength
num_classes = 2

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=64, epochs=15,
          validation_data=(X_dev, y_dev))

