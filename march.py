import pandas as pd
from processing import read, process
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, TimeDistributed, ConvLSTM2D
from keras.utils import to_categorical
from keras import initializers, optimizers

def evaluate_LSTM(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 25, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(
		50, 
		input_shape=(n_timesteps, n_features), 
		kernel_initializer=initializers.GlorotUniform(), 
		bias_initializer=initializers.Zeros(),
		recurrent_dropout=0.3,
		dropout=0.4
	))
	model.add(Dense(
		25, 
		activation='selu',
		kernel_initializer=initializers.GlorotUniform(), 
		bias_initializer=initializers.TruncatedNormal(mean=1, stddev=0.3)
	))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.7), metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy


def evaluate_CNN_LSTM(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = -1, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(10, recurrent_dropout=0.3, dropout=0.4))
	model.add(Dropout(0.3))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.4), metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

def evaluate_ConvLSTM2D(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape into subsequences (samples, time steps, rows, cols, channels)
	n_steps, n_length = -1, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(trainX.shape[1], 1, n_length, n_features)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.4), metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
 
def prep_dataset(rootdir: str, cats):
	cats = read.categorize(read.listdirs(rootdir), cats)
	Y, X, i = np.empty(0), np.empty((0, 64, 224)), 0
	for c in cats:
		csi = read.getCSI(cats[c][0].path)
		csi = process.extractAm(csi.reshape(csi.shape[0], -1))
		csi = process.filter1dUniform(csi, 15, 0)
		csi = process.norm(csi)
		csi = process.to_timeseries(csi)
		Y = np.concatenate([Y, np.tile([i], csi.shape[0])])
		X = np.concatenate([X, csi])
		i += 1
	
	cats = to_categorical(Y)
	return X, cats

cats = ['bottle', 'empty']
train_x, train_y = prep_dataset('./csidata/1_distortion_objects/1', cats)
test_x, test_y = prep_dataset('./csidata/1_distortion_objects/2', cats)

for i in range(10):
	print(i, '\n-----Точность LSTM:', evaluate_LSTM(train_x ,train_y, test_x, test_y))
	# print(i, '\n-----Точность CNN_LSTM:', evaluate_CNN_LSTM(train_x ,train_y, test_x, test_y))
	# print(i, '\n-----Точность ConvLSTM2D:', evaluate_ConvLSTM2D(train_x ,train_y, test_x, test_y))