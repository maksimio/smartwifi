import numpy as np
from sklearn import metrics
from processing import read, process, ml
from keras.utils import to_categorical
np.set_printoptions(edgeitems=30, linewidth=10000)

def prep_dataset_class_label(rootdir: str, files, cats):
	files = read.categorize(read.listdirs(rootdir), files)
	Y_mc, Y_ml, X, i = np.empty(0), np.empty((0, len(cats))), np.empty((0, 128, 224)), 0
	
	for f in files:
		csi = read.getCSI(files[f][0].path)
		csi = process.extractAm(csi.reshape(csi.shape[0], -1, order='F'))
		csi = process.filter1dUniform(csi, 15, 0)
		csi = process.norm(csi)
		csi = process.to_timeseries(csi)
		Y_mc = np.concatenate([Y_mc, np.tile([i], csi.shape[0])])
		Y_ml = np.concatenate([Y_ml, np.tile([c in f for c in cats], (csi.shape[0], 1))])
		X = np.concatenate([X, csi])
		i += 1
	
	return X, to_categorical(Y_mc), Y_ml

def print_metrics_ML(prefix: str, accuracy, test, y_pred):
	pred = y_pred > 0.5
	micro = metrics.f1_score(test, pred, average='micro')
	macro = metrics.f1_score(test, pred, average='macro')
	print(prefix + '_ML: accuracy {:.2f}%, F1-MICRO {:.2f}%, macro {:.2f}%'.format(accuracy, micro, macro))

def print_metrics_MC(prefix: str, accuracy, y_test, y_pred):
	test = np.argmax(y_test, axis=1)+1
	pred = np.argmax(y_pred, axis=1)+1
	micro = metrics.f1_score(test, pred, average='micro')
	macro = metrics.f1_score(test, pred, average='macro')
	print(prefix + '_MC: accuracy {:.2f}%, F1-MICRO {:.2f}%, macro {:.2f}%'.format(accuracy, micro, macro))

def ml_single(fn, train_x ,train_y_ml, test_x, test_y_ml, *args):
	res = np.empty((0, test_y_ml.shape[0]))
	for i in range(len(cats)):
		train = to_categorical(train_y_ml[:, i])
		test = to_categorical(test_y_ml[:, i])
		_, y_pred = fn(train_x ,train, test_x, test, *args)
		result = y_pred[:, 1]
		result = np.reshape(result,(1, result.size))
		res = np.concatenate([res, result])

	return 0, res.T

files = [
'-air.dat',
'-bottle.dat',
'-vaze.dat',
'-metal.dat',
'-metal_vaze.dat',
'-bottle_metal.dat',
'-bottle_vaze.dat',
'-bottle_metal_vaze.dat',
]
cats = ['bottle', 'vaze', 'metal']

train_x, train_y_mc, train_y_ml = prep_dataset_class_label('./csidata/2_multiple/5/train', files, cats)
test_x, test_y_mc, test_y_ml = prep_dataset_class_label('./csidata/2_multiple/5/test', files, cats)


for i in range(10): # l - label, c - class, s - single
	accuracy, y_pred = ml_single(ml.my_LSTM, train_x ,train_y_ml, test_x, test_y_ml, False)
	print_metrics_ML(str(i) + ') ---LSTM-S', accuracy, test_y_ml, y_pred)
	# accuracy, y_pred = ml.my_LSTM(train_x ,train_y_ml, test_x, test_y_ml, True)
	# print_metrics_ML(str(i) + ') ---LSTM-L', accuracy, test_y_ml, y_pred)
	# accuracy, y_pred = ml.my_LSTM(train_x ,train_y_mc, test_x, test_y_mc, False)
	# print_metrics_MC(str(i) + ') ---LSTM-C', accuracy, test_y_mc, y_pred)

	accuracy, y_pred = ml_single(ml.CNN_LSTM, train_x ,train_y_ml, test_x, test_y_ml, False)
	print_metrics_ML(str(i) + ') ---CNN_LSTM-S', accuracy, test_y_ml, y_pred)
	# accuracy, y_pred = ml.CNN_LSTM(train_x ,train_y_ml, test_x, test_y_ml, True)
	# print_metrics_ML(str(i) + ') ---CNN_LSTM-L', accuracy, test_y_ml, y_pred)
	# accuracy, y_pred = ml.CNN_LSTM(train_x ,train_y_mc, test_x, test_y_mc, False)
	# print_metrics_MC(str(i) + ') ---CNN_LSTM-C', accuracy, test_y_mc, y_pred)

	accuracy, y_pred = ml_single(ml.Conv_LSTM2D, train_x ,train_y_ml, test_x, test_y_ml, False)
	print_metrics_ML(str(i) + ') ---ConvLSTM2D-S', accuracy, test_y_ml, y_pred)
	# accuracy, y_pred = ml.Conv_LSTM2D(train_x ,train_y_ml, test_x, test_y_ml, True)
	# print_metrics_ML(str(i) + ') ---ConvLSTM2D-L', accuracy, test_y_ml, y_pred)
	# accuracy, y_pred = ml.Conv_LSTM2D(train_x ,train_y_mc, test_x, test_y_mc, False)
	# print_metrics_MC(str(i) + ') ---ConvLSTM2D-C', accuracy, test_y_mc, y_pred)

	accuracy, y_pred = ml_single(ml.CNN, train_x ,train_y_ml, test_x, test_y_ml, False, True)
	print_metrics_ML(str(i) + ') ---CNN-S', accuracy, test_y_ml, y_pred)
	# accuracy, y_pred = ml.CNN(train_x ,train_y_ml, test_x, test_y_ml, True, True)
	# print_metrics_ML(str(i) + ') ---CNN-L', accuracy, test_y_ml, y_pred)
	# accuracy, y_pred = ml.CNN(train_x ,train_y_mc, test_x, test_y_mc, False, True)
	# print_metrics_MC(str(i) + ') ---CNN-C', accuracy, test_y_mc, y_pred)

	print()

