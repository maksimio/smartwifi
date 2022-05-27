'''Функции визуализации CSI'''
from matplotlib import pyplot as plt
import numpy as np
import cv2

def imsave(fpath: str, csi: np.ndarray, maxHeight=500) -> None:
  csi = csi[:maxHeight]
  csi =  csi / csi.max() * 255
  cv2.imwrite(fpath, csi)

def imshow() -> None:
    ...

def plot() -> None:
    ...
