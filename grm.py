from processing import read, process, view
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

# 1. Чтение данных и разбиение на 2 канала
rootdir = './csidata/1_distortion_objects/1'
cats = read.categorize(read.listdirs(rootdir), ['bottle', 'empty'])
csi = read.getCSI(cats['bottle'][0].path)

csi = process.reshape224x1(csi)
am = process.extractAm(csi)
ph = process.extractPh(csi)

# 2. Сглаживание
tl = process.swap2axes(am)
car = process.filter1dUniform(tl, 20, 1)
gauss = process.filter1dGauss(tl, 20, 1)
car = process.diff1d(gauss, axis=1)

# 3. Графики
plt.plot(car[25])
plt.plot(gauss[25])
plt.show()

# 4. Автокоррелограмма
view.autocorr(am[42])

# 5. Спектр numpy
f = np.abs(np.fft.fft(am, axis=1))
plt.plot(f[2])
plt.show()

# 5. Спектр scipy
f, t, Sxx = signal.spectrogram(tl[3])
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()