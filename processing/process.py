import numpy as np
from scipy import ndimage

# Преобразование к виду 56 x 4 - амплитуды или фазы
def reshape4x56(csi: np.ndarray) -> np.ndarray:
    csi = np.reshape(csi, (csi.shape[0], csi.shape[1], -1))
    return np.transpose(csi, (0, 2, 1))

# Преобразование к виду 224 x 1 - амплитуды или фазы
def reshape224x1(csi: np.ndarray) -> np.ndarray:
    csi = reshape4x56(csi)
    return np.reshape(csi, (csi.shape[0], -1))

def to_timeseries(dataset, split_len = 64, step = 15):
	'''Преобразует dataset, где первая размерность - отсчеты во времени
	в датасет, где первая размерность - индексы для групп отсчетов во времени.
	Увеличивает shape на 1. split_len - длина временной последовательности, step - шаг выборки временных последовательностей.
	В случае, когда step >= split_len, выбранные последовательности не имеют пересечений'''
	conc = np.ndarray((0, split_len, *dataset.shape[1:]))
	count = dataset.shape[0]
	for i, j in zip(range(0, count - split_len, step), range(split_len, count, step)):
		d = dataset[i:j]
		conc = np.concatenate([conc, d.reshape(1, *d.shape)])
	return conc

def extractAm(csi: np.ndarray) -> np.ndarray:
    return np.abs(csi)

def extractPh(csi: np.ndarray, unwrap=False) -> np.ndarray:
    return np.angle(csi)

# def unwrapPh(ph: np.ndarray, axis=0) -> np.ndarray:
  # Нужно доработать, разделяя массив
#     '''Фазы выпрямляются при переходе через границу 2PI'''
#     return np.unwrap(ph, axis=axis)

# def down(csi: np.ndarray) -> np.ndarray:
#     return csi - csi.min()

def swap2axes(csi: np.ndarray) -> np.ndarray:
    return csi.swapaxes(1, 0)

def filter1dGauss(arr: np.ndarray, sigma=10, axis=-1) -> np.ndarray:
    '''Фильтр гаусса'''
    return ndimage.gaussian_filter1d(arr, sigma, axis)

def filter1dUniform(arr: np.ndarray, size=10, axis=-1) -> np.ndarray:
    '''Фильтр скользящего среднего'''
    return ndimage.uniform_filter1d(arr, size, axis)

def diff1d(arr: np.array, n=1, axis=-1) -> np.ndarray:
    '''Производная порядка n'''
    return np.diff(arr, n=n, axis=axis)

def splitLen(arr: np.array, count: int) -> list:
    split = np.array_split(arr, len)
    print(split[-2].shape)
    return 