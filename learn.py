import numpy as np
from sklearn import metrics
from processing import read, process, ml
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


np.set_printoptions(edgeitems=30, linewidth=10000)

SPLIT_LEN = 64
STEP = 64

def prep_dataset_class_label(rootdir: str, files, cats):
  files = read.categorize(read.listdirs(rootdir), files)
  Y_mc, Y_ml, X, i = np.empty(0), np.empty((0, len(cats))), np.empty((0, SPLIT_LEN, 224)), 0
  
  for f in files:
    csi = read.getCSI(files[f][0].path)
    csi = process.extractAm(csi.reshape(csi.shape[0], -1, order='F'))
    csi = process.filter1dUniform(csi, 5, 0)
    # csi = process.norm(csi)
    csi = process.to_timeseries(csi, split_len=SPLIT_LEN, step=STEP)
    # -----------------------------------------------
    
    plt.rcParams["figure.autolayout"] = True
    ax = plt.axes(projection='3d')
    z = csi[3,:,:]
    print('zz', z.shape)
    y = np.arange(len(z))
    x = np.arange(len(z[0]))
    (x ,y) = np.meshgrid(x,y)
    ax.plot_surface(y,x,z, edgecolor='green', lw=0.2, rstride=2, cstride=4, alpha=0.3)
    ax.view_init(30, -45, 0)
    ax.set_xlabel('№ пакета')
    ax.set_ylabel('№ поднесущей')
    ax.set_zlabel('Амплитуда, мВт')
    plt.savefig('kkk2.png', format='png', dpi=600)
    print(csi.shape)

    # ------------------------------------------------
    exit()

    Y_mc = np.concatenate([Y_mc, np.tile([i], csi.shape[0])])
    Y_ml = np.concatenate([Y_ml, np.tile([c in f for c in cats], (csi.shape[0], 1))])
    X = np.concatenate([X, csi])
    i += 1
  
  return X, to_categorical(Y_mc), Y_ml

def convert_mc2ml(mc):
  '''Конвертация multiclass ответов в multilabel'''
  a = np.argmax(mc, axis=1)
  m = np.ndarray((a.shape[0], len(cats)))
  m.fill(0)
  for i in range(m.shape[0]):
    metal = a[i] == 3 or a[i] == 4 or a[i] == 5 or a[i] == 7
    bottle = a[i] == 1 or a[i] == 5 or a[i] == 6 or a[i] == 7
    vaze = a[i] == 2 or a[i] == 4 or a[i] == 6 or a[i] == 7
    m[i] = [bottle, vaze, metal]

  return m.astype(int)

def print_metrics(prefix: str, accuracy, test, y_pred):
  '''Для multilabel классификации'''
  pred = y_pred > 0.5
  f1 = metrics.f1_score(test, pred, average='micro')
  print(prefix + '_: F1-MICRO {:.2f}%, accuracy {:.2f}%'.format(f1, accuracy))

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
  # accuracy, y_pred = ml_single(ml.my_LSTM, train_x ,train_y_ml, test_x, test_y_ml, False)
  # print_metrics(str(i) + ') ---LSTM-S', accuracy, test_y_ml, y_pred)
  accuracy, y_pred = ml.my_LSTM(train_x ,train_y_ml, test_x, test_y_ml, True)
  print_metrics(str(i) + ') ---LSTM-L', accuracy, test_y_ml, y_pred)
  # accuracy, y_pred = ml.my_LSTM(train_x ,train_y_mc, test_x, test_y_mc, False)
  # print_metrics(str(i) + ') ---LSTM-C', accuracy, test_y_ml, convert_mc2ml(y_pred))

  # accuracy, y_pred = ml_single(ml.CNN_LSTM, train_x ,train_y_ml, test_x, test_y_ml, False)
  # print_metrics(str(i) + ') ---CNN_LSTM-S', accuracy, test_y_ml, y_pred)
  # accuracy, y_pred = ml.CNN_LSTM(train_x ,train_y_ml, test_x, test_y_ml, True)
  # print_metrics(str(i) + ') ---CNN_LSTM-L', accuracy, test_y_ml, y_pred)
  # accuracy, y_pred = ml.CNN_LSTM(train_x ,train_y_mc, test_x, test_y_mc, False)
  # print_metrics(str(i) + ') ---CNN_LSTM-C', accuracy, test_y_ml, convert_mc2ml(y_pred))

  # accuracy, y_pred = ml_single(ml.Conv_LSTM2D, train_x ,train_y_ml, test_x, test_y_ml, False)
  # print_metrics(str(i) + ') ---ConvLSTM2D-S', accuracy, test_y_ml, y_pred)
  # accuracy, y_pred = ml.Conv_LSTM2D(train_x ,train_y_ml, test_x, test_y_ml, True)
  # print_metrics(str(i) + ') ---ConvLSTM2D-L', accuracy, test_y_ml, y_pred)
  # accuracy, y_pred = ml.Conv_LSTM2D(train_x ,train_y_mc, test_x, test_y_mc, False)
  # print_metrics(str(i) + ') ---ConvLSTM2D-C', accuracy, test_y_ml, convert_mc2ml(y_pred))

  # accuracy, y_pred = ml_single(ml.CNN, train_x ,train_y_ml, test_x, test_y_ml, False, True)
  # print_metrics(str(i) + ') ---CNN-S', accuracy, test_y_ml, y_pred)
  # accuracy, y_pred = ml.CNN(train_x ,train_y_ml, test_x, test_y_ml, True, True)
  # print_metrics(str(i) + ') ---CNN-L', accuracy, test_y_ml, y_pred)
  # accuracy, y_pred = ml.CNN(train_x ,train_y_mc, test_x, test_y_mc, False, True)
  # print_metrics(str(i) + ') ---CNN-C', accuracy, test_y_ml, convert_mc2ml(y_pred))
  # print()

