import numpy as np
import pandas as pd
from processing import process, read


def df_from_file(filepath):
  csi = read.getCSI(filepath)
  am = process.extractAm(csi)
  am = process.reshape224x1(am)
  df = pd.DataFrame(am)
  df['target'] = filepath.split('/')[-1].split('.')[0]
  return df


object_names = ['empty', 'bottle', 'casserole', 'grater']
dfs = []
for on in object_names:
  filepath = './csidata/1_distortion_objects/1/' + on + '.dat'
  df = df_from_file(filepath)
  dfs.append(df)

df = pd.concat(dfs)
# print(df)
df.to_csv('out.csv', index=False)

df_lst = []
for i in range(int(df.shape[1] / 56)):
  item = df[[k + i*56 for k in range(0, 56)]]
  newDf = pd.DataFrame()
  newDf['std_' + str(i)] = item.std(axis=1)
  # newDf['ymax_' + str(i)] = pd.DataFrame(np.fft.fft(item.sub(item.mean(axis=1)) - 1, axis=1)).abs().max() ** 2 / df.shape[1]
  newDf['mu42_' + str(i)] = (item ** 4).mean(axis=1) / ((item ** 2).mean(axis=1) ** 4)
  newDf['ln_' + str(i)] = np.log(item.mean(axis=1))
  newDf['mu32_' + str(i)] = (item ** 3).mean(axis=1) / ((item ** 2).mean(axis=1) ** (3 / 2))
  newDf['skew_' + str(i)] = item.skew(axis=1)
  newDf['kurt_' + str(i)] = item.kurt(axis=1)
  df_lst.append(newDf)

df_lst.append(df['target'])
newDf = pd.concat(df_lst, axis=1)
print(newDf)
newDf.to_csv('out2.csv', index=False)

  # self.__df_csi_lst = []
  # for i in range(int(df.shape[1] / self.__num_tones)):
  #     item = df[[k + i*self.__num_tones for k in range(0, self.__num_tones)]]
  #     self.__df_csi_lst.append(item)

        # i = 1
        # for df_part in self.__df_csi_lst:
        #     df['std_' + str(i)] = df_part.std(axis=1)
        #     # df['ymax_' + str(i)] = pd.DataFrame(np.fft.fft(df_part.sub(df_part.mean(axis=1)) - 1, axis=1)).abs().max() ** 2 / df.shape[1]
        #     df['mu42_' + str(i)] = (df_part ** 4).mean(axis=1) / ((df_part ** 2).mean(axis=1) ** 4)
        #     df['ln_' + str(i)] = np.log(df_part.mean(axis=1))
        #     df['mu32_' + str(i)] = (df_part ** 3).mean(axis=1) / ((df_part ** 2).mean(axis=1) ** (3 / 2))
        #     df['skew_' + str(i)] = df_part.skew(axis=1)
        #     df['kurt_' + str(i)] = df_part.kurt(axis=1)
        #     i += 1
        