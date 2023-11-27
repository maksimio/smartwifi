from processing import read, process, view
from matplotlib import pyplot as plt
from scipy import signal

# 1. Чтение данных и разбиение на 2 канала
rootdir = './csidata/1_distortion_objects/1'
cats = read.categorize(read.listdirs(rootdir), ['bottle', 'empty'])
csi = read.getCSI(cats['bottle'][0].path)
csi = process.reshape224x1(csi)

am = process.extractAm(csi)
ph = process.extractPh(csi)

# 2. Сглаживание
timeLine = process.swap2axes(am)[25]
print(timeLine.shape)
timeLine = process.moving_average(timeLine, 50)

plt.plot(timeLine)
plt.show()

# 3. Графики
# print(ph.shape)
# ph = process.swap2axes(ph)
# view.plotLine(ph)

# 4. Автокоррелограмма
corr = process.autocorr(timeLine)
plt.plot(corr)
plt.show()

# 5. Спектрограмма
f, t, Sxx = signal.spectrogram(timeLine, fs=100000)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()