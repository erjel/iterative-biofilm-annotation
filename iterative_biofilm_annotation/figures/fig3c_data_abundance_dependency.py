# based on cellpose_data_dependence_plotting_v1.ipynb

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import re
import pandas as pd
import matplotlib as mpl

import yaml

import sys
sys.path.append(str(Path(r'D:/Users/Eric/src/stardist_mpcdf')))  

from stardist_mpcdf.data import readDataset


with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

datasetname = 'full_semimanual-raw'
outputname = 'test.png'

final_models = config['cellpose_models_raw_full_low']
final_models

def get_minor_models(modelname):
    
    tmp = modelname.split('_ep')[-1].split('_dep')
    epochs = int(tmp[0])
    delta_epochs = int(tmp[-1])
    minor_models = []
    
    for ep in range(epochs, 1, -delta_epochs):
        minor_models.append(modelname.replace('_ep500', f'_ep{ep}'))
        
    return minor_models

modelname = 'cellpose_patches-semimanual-raw-64x128x128_True_25prc_rep1_ep500_dep125'

models = []
for modelname in final_models:
    for m in get_minor_models(modelname):
        models.append(m)

accuracy_files = [f'data/{m}/accuracy_manual_raw_v3.csv' for m in models]
#accuracy_files = [f'data/{m}/accuracy_full_semimanual-raw.csv' for m in models]
accuracy_files = [Path(f).parent for f in accuracy_files if Path(f).is_file()]

accuracy_files

df = pd.DataFrame(columns=['path', 'type', 'percentage', 'replicate', 'epoch', 'cell_number', 'accuracy_manual', 'accuracy_semimanual'])

p = '.*True_(?P<percentage>[\d\.]+)prc_rep(?P<replicate>\d+)_ep(?P<epoch>\d+)_dep.*'
pattern = re.compile(p)

for f in accuracy_files:
    match = pattern.match(str(f))
    df = df.append({'path':str(f) , 'type':'cellpose', **match.groupdict()}, ignore_index=True)

stardist_models = config['stardist_models_dependency'] # stardist_models_raw


accuracy_files = [f'data/{m}/accuracy_full_semimanual-raw.csv' for m in stardist_models]
accuracy_files = [Path(f).parent for f in accuracy_files if Path(f).is_file()]

print(accuracy_files)


p = '.*True_(?P<percentage>[\d\.]+)prc_rep(?P<replicate>\d+)'
pattern = re.compile(p)

for f in accuracy_files:
    match = pattern.match(str(f))
    df = df.append({'path':str(f) , 'type':'stardist', 'epoch':500, **match.groupdict()}, ignore_index=True)

print(df)

Y = readDataset('patches-semimanual-raw-64x128x128')[1]

Y['test'] = []
Y['valid'] = []

for s in Y.keys():
    sum_Y = [np.sum(y) for y in Y[s]]
    Y[s] = [Y[s][i] for i in range(len(Y[s])) if sum_Y[i] > 0]

N_cells = [len(np.unique(y))-1 for y in Y['train']]

for index, row in df.iterrows():
    seed = int(row.replicate) if row.type == 'cellpose' else 42
    rng = np.random.RandomState(int(row.replicate))
    ind = rng.permutation(len(Y['train']))
    n_val = max(1, int(round(float(row.percentage) / 100 * len(ind))))
    df.iloc[index]['cell_number'] = np.sum([N_cells[i] for i in ind[:n_val]])
    
    for data_name, col in zip(['accuracy_manual_raw_v3.csv', 'accuracy_full_semimanual-raw.csv'], ['accuracy_manual', 'accuracy_semimanual']):
    #for data_name, col in zip(['accuracy_full_semimanual-raw.csv', 'accuracy_full_semimanual-raw.csv'], ['accuracy_manual', 'accuracy_semimanual']):
        if (Path(row.path) / data_name).is_file():
            data = np.genfromtxt(Path(row.path) / data_name, delimiter=' ')
            df.iloc[index][col] = data[1][np.where(data[0]==0.5)[0]][0]

        else:
            df.iloc[index][col] = np.nan

print(df)

print(df[df.type == 'cellpose'].replicate.unique())

df = df.astype({'accuracy_manual': 'float', 'accuracy_semimanual':'float', 'percentage':'float'})

print(df[df.epoch==500].groupby(['percentage', 'type'], as_index=False)['accuracy_manual', 'accuracy_semimanual'].mean())

#df_ = df[(df.epoch==500)].groupby(['percentage', 'type'], as_index=False)['accuracy_manual'].agg({'acc_std':'std', 'acc_mean':'mean'})
df_ = df[(df.epoch==500)].groupby(['percentage', 'type'], as_index=False)['accuracy_semimanual'].agg({'acc_std':'std', 'acc_mean':'mean'})

df_n = df[(df.epoch==500)].groupby(['percentage', 'type'], as_index=False).agg(lambda x: np.mean(x))

print(df_n)

f, ax1 = plt.subplots(1)


selection = (df_n.type == 'stardist')

ax1.errorbar(df_n[selection].cell_number, df_[selection]['acc_mean'], yerr=df_[selection]['acc_std'], label='stardist')
ax1.set_xlabel('Cell number')
ax1.set_ylabel('accuracy [a.u.]')

selection = (df_n.type == 'cellpose')

l = ax1.errorbar(df_n[selection].cell_number, df_[selection]['acc_mean'],
                 yerr=df_[selection]['acc_std'], ls='--')[0]
ax1.set_xlabel('Cell number')
ax1.set_ylabel('accuracy [a.u.]')


selection = (df_n.type == 'cellpose') & (df_n.percentage <= 25) & (np.logical_not(df_.acc_mean.isnull()))

print(df_[selection]['acc_mean'])

ax1.errorbar(df_n[selection].cell_number, df_[selection]['acc_mean'],
             yerr=df_[selection]['acc_std'], color=l.get_color(), ls='-',
             label='cellpose')
ax1.set_xlabel('Cell number')
ax1.set_ylabel('accuracy [a.u.]')
ax1.legend()
ax1.grid()


#for rep in range(1, 4):
#    df_rep = df[(df.percentage <= 25) & (df.replicate==rep) & (df.epoch == 500)]
#    ax.plot(df_rep.cell_number, df_rep.accuracy_semimanual)

plt.savefig('test.png')
