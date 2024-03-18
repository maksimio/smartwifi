import csiread
import numpy as np

# fpath = './csidata/1_distortion_objects/1/casserole.dat'
# fpath = './datatest/2_empty_re.dat'
fpath = './datatest/2_new_re.dat'
fpath = './datatest/fuck.dat'
fpath = './datatest/prig.dat'

# csidata = csiread.Atheros(None, bufsize=10, nrxnum=2, ntxnum=2)
# csidata.seek(fpath, 0, 10, endian='big')
# csidata.read(endian='big')
# print(csidata.csi.shape)


# exit()
data = csiread.Atheros(fpath, nrxnum=2, ntxnum=5, tones=56, if_report=True, pl_size=468)
print('перед чтением')
data.read(endian='big')
print('после чтения')
payload_len = np.bincount(data.payload_len).argmax()
csi = data.csi[(data.payload_len == payload_len) & (data.nc == 2)][:, :, :2, :2]
print(csi.shape)
