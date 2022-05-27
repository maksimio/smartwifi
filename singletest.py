import processing.process

import numpy as np


csi = processing.read.extractCSI('./csidata/1_metal_objects/1/bottle.dat')
print(csi)
csi = processing.process.extractAm(csi)
csi = processing.process.reshape4x56(csi)
# csi = np.diff(csi)
csi = np.reshape(csi, (csi.shape[0], -1))
# csi = processing.down(csi)
processing.view.imsave('./newtest4.jpg', csi)