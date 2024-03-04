import pandas as pd
from processing import read, process
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, TimeDistributed, ConvLSTM2D
from keras.utils import to_categorical

def evaluate_LSTM(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 2, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy


def evaluate_CNN_LSTM(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 2, 32
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
	model.add(LSTM(10))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

def evaluate_ConvLSTM2D(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 2, 32
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
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
 
def prep_dataset(rootdir):
	cats = read.categorize(read.listdirs(rootdir), ['bottle', 'empty'])

	csi = read.getCSI(cats['bottle'][0].path)
	csi = process.extractAm(csi.reshape(csi.shape[0], -1))
	csi = process.filter1dUniform(csi, 15, 0)
	csi = process.norm(csi)
	csi_b = process.to_timeseries(csi)

	csi = read.getCSI(cats['empty'][0].path)
	csi = process.extractAm(csi.reshape(csi.shape[0], -1))
	csi = process.filter1dUniform(csi, 15, 0)
	csi = process.norm(csi)
	csi_e = process.to_timeseries(csi)

	y_b = np.tile(np.array([0, 1]), (csi_b.shape[0], 1))
	y_e = np.tile(np.array([1, 0]), (csi_e.shape[0], 1))
	return np.concatenate([csi_b, csi_e]), np.concatenate([y_b, y_e])


train_x, train_y = prep_dataset('./csidata/1_distortion_objects/1')
test_x, test_y = prep_dataset('./csidata/1_distortion_objects/2')
df = pd.DataFrame(test_y)
print('Не работает распознавание:', df[1].sum() / df.shape[0])


np.random.seed = 42
for i in range(10):
	print(i, '-----Точность LSTM:', evaluate_LSTM(train_x ,train_y, test_x, test_y))
	print(i, '-----Точность CNN_LSTM:', evaluate_CNN_LSTM(train_x ,train_y, test_x, test_y))
	print(i, '-----Точность ConvLSTM2D:', evaluate_ConvLSTM2D(train_x ,train_y, test_x, test_y))