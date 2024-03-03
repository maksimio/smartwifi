from processing import read, process
import numpy as np
 
rootdir = './csidata/1_distortion_objects/1'
def prep_dataset(rootdir):
	cats = read.categorize(read.listdirs(rootdir), ['bottle', 'empty'])
	csi_b = process.to_timeseries(read.getCSI(cats['bottle'][0].path))
	csi_e = process.to_timeseries(read.getCSI(cats['empty'][0].path))
	y_b = np.tile(np.array([0, 1]), (csi_b.shape[0], 1))
	y_e = np.tile(np.array([1, 0]), (csi_e.shape[0], 1))
	return np.concatenate([csi_b, csi_e]), np.concatenate([y_b, y_e])

# c = csi.reshape((csi.shape[0], -1))
# data = process.to_timeseries(csi, 1000, 500)

x, y = prep_dataset(rootdir)
print(x.shape,y)