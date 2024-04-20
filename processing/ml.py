from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, TimeDistributed, ConvLSTM2D, Conv2D, MaxPooling2D
from keras import initializers, optimizers, callbacks
import numpy as np

VERBOSE = 0
cb = callbacks.EarlyStopping(monitor='accuracy', patience=1, start_from_epoch=4)

def my_LSTM(trainX, trainy, testX, testy, isMultilabel: bool):
  verbose, epochs, batch_size = VERBOSE, 25, 16
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
  model = Sequential()
  model.add(LSTM(
    140, 
    input_shape=(n_timesteps, n_features), 
    kernel_initializer=initializers.GlorotUniform(), 
    bias_initializer=initializers.Zeros(),
    recurrent_dropout=0.2,
    dropout=0.3
  ))
  model.add(Dropout(0.7))
  model.add(Dense(
    100,
    activation='elu',
    kernel_initializer=initializers.GlorotUniform(), 
    bias_initializer=initializers.TruncatedNormal(mean=1, stddev=0.3)
  ))
  model.add(Dropout(0.5))

  if isMultilabel:
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(momentum=0.6), metrics=['accuracy'])
  else:
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.4), metrics=['accuracy'])
  
  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True, callbacks=[cb])
  _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)

  y_pred = model.predict(testX, verbose=verbose)
  return accuracy, y_pred

def CNN_LSTM(trainX, trainy, testX, testy, isMultilabel: bool):
  # define model
  verbose, epochs, batch_size = VERBOSE, 25, 16
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
  # reshape data into time steps of sub-sequences
  n_steps, n_length = -1, 4
  trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
  testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
  # define model
  model = Sequential()
  model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='elu'), input_shape=(None,n_length,n_features)))
  # model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
  model.add(TimeDistributed(Dropout(0.5)))
  model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
  model.add(TimeDistributed(Flatten()))
  model.add(LSTM(80, recurrent_dropout=0.3, dropout=0.4))
  model.add(Dropout(0.4))
  model.add(Dense(50, activation='relu'))
  
  if isMultilabel:
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(momentum=0.6), metrics=['accuracy'])
  else:
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.4), metrics=['accuracy'])

  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[cb])
  _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
  y_pred = model.predict(testX, verbose=verbose)
  return accuracy, y_pred

def Conv_LSTM2D(trainX, trainy, testX, testy, isMultilabel: bool):
  # define model
  verbose, epochs, batch_size = VERBOSE, 25, 16
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
  # reshape into subsequences (samples, time steps, rows, cols, channels)
  n_steps, n_length = -1, 4
  trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
  testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
  # define model
  model = Sequential()
  model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(trainX.shape[1], 1, n_length, n_features)))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100, activation='relu'))

  if isMultilabel:
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(momentum=0.6), metrics=['accuracy'])
  else:
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.4), metrics=['accuracy'])

  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[cb])
  _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
  y_pred = model.predict(testX, verbose=verbose)
  return accuracy, y_pred

def CNN(trainX, trainy, testX, testy, isMultilabel: bool, useChannels: bool):
  verbose, epochs, batch_size, channels = VERBOSE, 25, 16, 1

  if useChannels:
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 56, 4))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 56, 4))
    channels = 4

  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='elu', input_shape=(n_timesteps, n_features, channels)))
  # model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dropout(0.7))
  model.add(Dense(
    100,
    activation='elu',
    kernel_initializer=initializers.GlorotUniform(), 
    bias_initializer=initializers.TruncatedNormal(mean=1, stddev=0.3)
  ))
  model.add(Dropout(0.5))

  if isMultilabel:
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(momentum=0.6), metrics=['accuracy'])
  else:
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.4), metrics=['accuracy'])

  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[cb])
  _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
  y_pred = model.predict(testX, verbose=verbose)
  return accuracy, y_pred