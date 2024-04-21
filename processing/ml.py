from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten, TimeDistributed, ConvLSTM2D, Conv2D, MaxPooling2D
from keras import initializers, optimizers, callbacks
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

VERBOSE = 0
EPOCHS = 25
BATCH_SIZE = 128
cb = callbacks.EarlyStopping(monitor='accuracy', patience=1, start_from_epoch=4)

def RNN(trainX, trainy, testX, testy, isMultilabel: bool):
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
  
  history = model.fit(trainX, trainy, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=True, callbacks=[cb])
  # plt_x = [*range(1, epochs + 1)]
  # plt.plot(plt_x, history.history['accuracy'], marker='o', color='m')
  # plt.plot(plt_x, history.history['loss'], marker='x', color='g')
  # plt.ylabel('Показатель')
  # plt.xlabel('№ эпохи обучения')
  # plt.grid()
  # plt.legend(['Точность (accuracy)', 'Потери (loss)'], loc='lower left')
  # plt.savefig('out1.png')
  # exit()
  
  _, accuracy = model.evaluate(testX, testy, batch_size=BATCH_SIZE, verbose=VERBOSE)

  y_pred = model.predict(testX, verbose=VERBOSE)
  return accuracy, y_pred

def CRNN(trainX, trainy, testX, testy, isMultilabel: bool, useChannels: bool):
  channels = 1
  n_steps, n_length = -1, 8
  
  if useChannels:
    trainX = np.reshape(trainX, (trainX.shape[0], n_steps, n_length, 56, 4))
    testX = np.reshape(testX, (testX.shape[0], n_steps, n_length, 56, 4))
    channels = 4

  n_timesteps, n_features, n_outputs = trainX.shape[2], trainX.shape[3], trainy.shape[1]

  model = Sequential()
  model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='elu', input_shape=(n_timesteps, n_features, channels))))
  model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
  model.add(TimeDistributed(Flatten()))
  model.add(Dropout(0.6))
  model.add(LSTM(80, recurrent_dropout=0.3, dropout=0.4))
  model.add(Dropout(0.4))
  model.add(Dense(50, activation='relu'))
  
  if isMultilabel:
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(momentum=0.6), metrics=['accuracy'])
  else:
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.4), metrics=['accuracy'])

  model.fit(trainX, trainy, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[cb])
  _, accuracy = model.evaluate(testX, testy, batch_size=BATCH_SIZE, verbose=VERBOSE)
  y_pred = model.predict(testX, verbose=VERBOSE)
  return accuracy, y_pred

def CNN(trainX, trainy, testX, testy, isMultilabel: bool, useChannels: bool):
  channels = 1

  if useChannels:
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 56, 4))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 56, 4))
    channels = 4

  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='elu', input_shape=(n_timesteps, n_features, channels)))
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

  model.fit(trainX, trainy, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[cb])
  _, accuracy = model.evaluate(testX, testy, batch_size=BATCH_SIZE, verbose=VERBOSE)
  y_pred = model.predict(testX, verbose=VERBOSE)
  return accuracy, y_pred