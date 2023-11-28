'''Функции визуализации CSI'''
from matplotlib import pyplot as plt
import numpy as np
import cv2
from statsmodels.graphics.tsaplots import plot_acf


def imsave(fpath: str, csi: np.ndarray, maxHeight=500) -> None:
  csi = csi[:maxHeight]
  csi =  csi / csi.max() * 255
  cv2.imwrite(fpath, csi)

def imshow() -> None:
  ...

def plotLine(csi: np.ndarray, lineIndex=0) -> None:
  plt.plot(csi[lineIndex])
  plt.grid()
  plt.show()

def autocorr(arr1d: np.ndarray) -> None:
  plot_acf(arr1d)
  plt.grid()
  plt.show()