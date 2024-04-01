from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from processing import read, process
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, TimeDistributed, ConvLSTM2D
from keras.utils import to_categorical
from keras import initializers, optimizers
np.set_printoptions(edgeitems=30, linewidth=10000)


def evaluate_LSTM(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 1, 8, 16
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
	model.add(Dense(
		100, 
		activation='elu',
		kernel_initializer=initializers.GlorotUniform(), 
		bias_initializer=initializers.TruncatedNormal(mean=1, stddev=0.3)
	))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(momentum=0.6), metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

	y_pred = model.predict(testX)
	result = multilabel_confusion_matrix(testy, y_pred > 0.5)
	print(result)

	return accuracy

 
def prep_dataset(rootdir: str, files, cats):
	files = read.categorize(read.listdirs(rootdir), files)
	Y, X, i = np.empty((0, len(cats))), np.empty((0, 64, 224)), 0
	for f in files:
		csi = read.getCSI(files[f][0].path)
		csi = process.extractAm(csi.reshape(csi.shape[0], -1))
		# csi = process.filter1dUniform(csi, 15, 0)
		csi = process.norm(csi)
		csi = process.to_timeseries(csi)
		Y = np.concatenate([Y, np.tile([c in f for c in cats], (csi.shape[0], 1))])
		X = np.concatenate([X, csi])
		i += 1
	
	return X, Y

files = [
'-air.dat',
'-bottle.dat',
'-vaze.dat',
'-metal.dat',
# '-metal_vaze.dat',
# '-bottle_metal.dat',
# '-bottle_vaze.dat',
# '-bottle_metal_vaze.dat',
]
cats = ['bottle', 'vaze', 'metal']
train_x, train_y = prep_dataset('./csidata/2_multiple/5/train', files, cats)
test_x, test_y = prep_dataset('./csidata/2_multiple/5/test', files, cats)

for i in range(10):
	print(i, '\n-----Точность LSTM:', evaluate_LSTM(train_x ,train_y, test_x, test_y))