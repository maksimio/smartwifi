import numpy as np

# Преобразование к виду 56 x 4 - амплитуды или фазы
def reshape4x56(csi: np.ndarray) -> np.ndarray:
    csi = np.reshape(csi, (csi.shape[0], csi.shape[1], -1))
    return np.transpose(csi, (0, 2, 1))

# Преобразование к виду 224 x 1 - амплитуды или фазы
def reshape224x1(csi: np.ndarray) -> np.ndarray:
    csi = reshape4x56(csi)
    return np.reshape(csi, (csi.shape[0], -1))

# Нарезка по первой размерности по заданному чанку (по умолчанию - квадрат).
# Неполный чанк в конце отбрасывается
def chunks(csi: np.ndarray, height=None) -> np.ndarray:
    if height == None:
        height = csi.shape[0]

def extractAm(csi: np.ndarray) -> np.ndarray:
    return np.abs(csi)

def extractPh(csi: np.ndarray, unwrap=False) -> np.ndarray:
    '''Фазы выпрямляются при переходе через границу 2PI'''
    ph = np.angle(csi)
    if unwrap:
        axis = np.argwhere((np.array(csi.shape) == 56) | (np.array(csi.shape) == 114))[0][0]
        ph = np.unwrap(ph, axis=axis)
    return ph
    # return csiread.utils.calib(ph, k=csiread.utils.scidx(40, 1), axis=axis)


def down(csi: np.ndarray) -> np.ndarray:
    return csi - csi.min()

def swap2axes(csi: np.ndarray) -> np.ndarray:
    return csi.swapaxes(1, 0)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_average2(a, n=3):
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):] / np.max(result)