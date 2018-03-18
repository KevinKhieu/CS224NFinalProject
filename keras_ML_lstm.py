from keras.models import Sequential
from keras.layers import *
import numpy as np

from svm import getLSTMMultiData

print "Getting Data..."
X_train, y_train, X_dev, y_dev, X_test, y_test, maxLength, embeddingLength = getLSTMMultiData()

print "Building Model..."


data_dim = embeddingLength
timesteps = maxLength
num_classes = 2


print "LSTM Layers: 5,"
print "LSTM Layer Output Dimension: 64,"

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(64, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32
model.add(Dense(5, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=64, epochs=50,
          validation_data=(X_dev, y_dev))

pred = model.predict(np.array(X_test))