from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, TimeDistributed, ConvLSTM2D
from keras import initializers, optimizers

def multilabel_LSTM(trainX, trainy, testX, testy, isMultilabel: bool):
	verbose, epochs, batch_size = 0, 5, 16
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
	
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)

	y_pred = model.predict(testX)
	y_pred = y_pred > 0.5
	# result = multilabel_confusion_matrix(testy, y_pred)
	return accuracy, y_pred

def CNN_LSTM(trainX, trainy, testX, testy, isMultilabel: bool):
	# define model
	verbose, epochs, batch_size = 0, 5, 16
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

	if isMultilabel:
		model.add(Dense(n_outputs, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(momentum=0.6), metrics=['accuracy'])
	else:
		model.add(Dense(n_outputs, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.4), metrics=['accuracy'])

	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
	y_pred = model.predict(testX)
	y_pred = y_pred > 0.5
	return accuracy, y_pred