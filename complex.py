from processing import read, process, view
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt 
import numpy as np 
  
# create data of complex numbers 
data = np.array([1+2j, 2-4j, -2j, -4, 4+1j, 3+8j, -2-6j, 5]) 
  
# extract real part using numpy array 
x = data.real 
# extract imaginary part using numpy array 
y = data.real 
  
# plot the complex numbers 

# 1. Чтение данных и разбиение на 2 канала
rootdir = './csidata/1_distortion_objects/6'
cats = read.categorize(read.listdirs(rootdir), ['bottle', 'empty'])
csi = read.getCSI(cats['empty'][0].path)

# csi = process.reshape224x1(csi).transpose()
# csi = csi[100][:]

for i in range(100):
  csi1 = csi[i,:,1,0]
  plt.plot(csi1.real, csi1.imag, 'g*') 
  plt.ylabel('Imaginary') 
  plt.xlabel('Real') 
  plt.show() 
