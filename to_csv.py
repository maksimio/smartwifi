import numpy as np
import pandas as pd
from processing import process, read
from os.path import join

EXPERIMENT_DIR = './csidata/1_distortion_objects'
OUTPUT_DIR = 'out'
EXPERIMENTS = {'1': 'базовый', 
                 '2': 'с бутылкой сбоку',
                 '3': 'со стулом',
                 '4': 'дистанция 3 метра',
                 '5': 'в другом помещении',
                 '6': 'базовый - повтор'}
OBJECTS2 = {'empty': 'нет объекта',
            'bottle':'термос',
            'casserole':'кастрюля',
            'grater': 'терка'}

def df_from_file(filepath: str, experiment: str, target: str):
  csi = read.getCSI(filepath)
  am = process.extractAm(csi)
  am = process.reshape224x1(am)
  df = pd.DataFrame(am)
  df['experiment'] = experiment
  df['target'] = target
  return df

dfs = []
for exp_name, exp_description in EXPERIMENTS.items():
  for obj_name, obj_description in OBJECTS2.items():
    filepath = join(EXPERIMENT_DIR, exp_name, obj_name + '.dat')
    df = df_from_file(filepath, exp_description, obj_description)
    dfs.append(df)

df = pd.concat(dfs)
df.to_csv(join(OUTPUT_DIR, 'out.csv'), index=False)

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

df_lst.append(df['experiment'])
df_lst.append(df['target'])
newDf = pd.concat(df_lst, axis=1)
print(newDf)
newDf.to_csv(join(OUTPUT_DIR, 'out-признаки.csv'), index=False)
