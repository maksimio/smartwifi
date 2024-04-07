import numpy as np
from sklearn import metrics
from processing import read, process, ml
from keras.utils import to_categorical
np.set_printoptions(edgeitems=30, linewidth=10000)
 
def prep_dataset_multiclass(rootdir: str, cats):
	cats = read.categorize(read.listdirs(rootdir), cats)
	Y, X, i = np.empty(0), np.empty((0, 128, 224)), 0
	for c in cats:
		csi = read.getCSI(cats[c][0].path)
		csi = process.extractAm(csi.reshape(csi.shape[0], -1, order='F'))
		csi = process.filter1dUniform(csi, 15, 0)
		csi = process.norm(csi)
		csi = process.to_timeseries(csi)
		Y = np.concatenate([Y, np.tile([i], csi.shape[0])])
		X = np.concatenate([X, csi])
		i += 1
	
	cats = to_categorical(Y)
	return X, cats

def prep_dataset_multilabel(rootdir: str, files, cats):
	files = read.categorize(read.listdirs(rootdir), files)
	Y, X, i = np.empty((0, len(cats))), np.empty((0, 128, 224)), 0
	for f in files:
		csi = read.getCSI(files[f][0].path)
		csi = process.extractAm(csi.reshape(csi.shape[0], -1, order='F'))
		csi = process.filter1dUniform(csi, 15, 0)
		csi = process.norm(csi)
		csi = process.to_timeseries(csi)
		Y = np.concatenate([Y, np.tile([c in f for c in cats], (csi.shape[0], 1))])
		X = np.concatenate([X, csi])
		i += 1
	
	return X, Y

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
train_x_ml, train_y_ml = prep_dataset_multilabel('./csidata/2_multiple/5/train', files, cats)
test_x_ml, test_y_ml = prep_dataset_multilabel('./csidata/2_multiple/5/test', files, cats)

train_x_mc, train_y_mc = prep_dataset_multiclass('./csidata/2_multiple/5/train', files)
test_x_mc, test_y_mc = prep_dataset_multiclass('./csidata/2_multiple/5/test', files)

for i in range(10):
	accuracy, y_pred = ml.my_LSTM(train_x_ml ,train_y_ml, test_x_ml, test_y_ml, True)
	print_metrics_ML(str(i) + ') ---LSTM', accuracy, test_y_ml, y_pred)
	
	accuracy, y_pred = ml.my_LSTM(train_x_mc ,train_y_mc, test_x_mc, test_y_mc, False)
	print_metrics_MC(str(i) + ') ---LSTM', accuracy, test_y_mc, y_pred)

	accuracy, y_pred = ml.CNN_LSTM(train_x_ml ,train_y_ml, test_x_ml, test_y_ml, True)
	print_metrics_ML(str(i) + ') ---CNN_LSTM', accuracy, test_y_ml, y_pred)

	accuracy, y_pred = ml.CNN_LSTM(train_x_mc ,train_y_mc, test_x_mc, test_y_mc, False)
	print_metrics_MC(str(i) + ') ---CNN_LSTM', accuracy, test_y_mc, y_pred)

	accuracy, y_pred = ml.Conv_LSTM2D(train_x_ml ,train_y_ml, test_x_ml, test_y_ml, True)
	print_metrics_ML(str(i) + ') ---ConvLSTM2D', accuracy, test_y_ml, y_pred)

	accuracy, y_pred = ml.Conv_LSTM2D(train_x_mc ,train_y_mc, test_x_mc, test_y_mc, False)
	print_metrics_MC(str(i) + ') ---ConvLSTM2D', accuracy, test_y_mc, y_pred)

	accuracy, y_pred = ml.CNN(train_x_ml ,train_y_ml, test_x_ml, test_y_ml, True, True)
	print_metrics_ML(str(i) + ') ---CNN', accuracy, test_y_ml, y_pred)

	accuracy, y_pred = ml.CNN(train_x_mc ,train_y_mc, test_x_mc, test_y_mc, False, True)
	print_metrics_MC(str(i) + ') ---CNN', accuracy, test_y_mc, y_pred)

	print()

