from argparse import ArgumentParser, Namespace
import logging
import os
import re
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib as mpl
from tifffile import imread

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def get_minor_models(accuracy_path: Path) -> List[Path]:
    modelname = accuracy_path.parent.name
    tmp = modelname.split('_ep')[-1].split('_dep')
    epochs = int(tmp[0])
    delta_epochs = int(tmp[-1])

    minor_models = []
    
    for ep in range(epochs, 1, -delta_epochs):
        minor_models.append(accuracy_path.parent.parent / modelname.replace('_ep500', f'_ep{ep}') / accuracy_path.name)
        
    return minor_models


def plot_data_abundance_vs_accuracy(
    output_dir: Path,
    training_data_stardist: Path,
    cellpose_accuracies: List[Path],
    stardist_accuracies: List[Path],
    ) -> None:

    accuracy_files = []
    for cellpose_accuracy in cellpose_accuracies:
        for m in get_minor_models(cellpose_accuracy):
            accuracy_files.append(m)


    cellpose_model_paths = [Path(f).parent for f in accuracy_files if Path(f).is_file()]


    df = pd.DataFrame(columns=['path', 'type', 'percentage', 'replicate', 'epoch', 'cell_number', 'accuracy_manual', 'accuracy_semimanual'])

    p = '.*True_(?P<percentage>[\d\.]+)prc_rep(?P<replicate>\d+)_ep(?P<epoch>\d+)_dep.*'
    pattern = re.compile(p)

    for f in cellpose_model_paths:
        match = pattern.match(str(f))
        df = df.append({'path':str(f) , 'type':'cellpose', **match.groupdict()}, ignore_index=True)

    accuracy_files = stardist_accuracies
    accuracy_files = [Path(f).parent for f in accuracy_files if Path(f).is_file()]


    p = '.*True_(?P<percentage>[\d\.]+)prc_rep(?P<replicate>\d+)'
    pattern = re.compile(p)

    for f in accuracy_files:
        match = pattern.match(str(f))
        df = df.append({'path':str(f) , 'type':'stardist', 'epoch':500, **match.groupdict()}, ignore_index=True)

    Y = {
        'train':
            list(map(imread, training_data_stardist.glob('train/images/*.tif')))
    }
    logger.info(f'Training data: {training_data_stardist}')
    logger.info(f"Number of training images: {len(Y['train'])}")

    # TODO(erjel): Refactor this pandas chaos!
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

    logger.info(f"Unique cellpose replicate numbers: {df[df.type == 'cellpose'].replicate.unique()}")

    df = df.astype({'accuracy_manual': 'float', 'accuracy_semimanual':'float', 'percentage':'float', 'epoch':'int'})

    logger.info(f'Pre-filtering model types: {df.type.unique()}')
    logger.info(f"Pre-filtering cellpose selection:\n{df[(df.type=='cellpose') & (df.epoch==500)]}")

    print(df[df.epoch==500].groupby(['percentage', 'type'], as_index=False)['accuracy_manual', 'accuracy_semimanual'].mean())

    #df_ = df[(df.epoch==500)].groupby(['percentage', 'type'], as_index=False)['accuracy_manual'].agg({'acc_std':'std', 'acc_mean':'mean'})
    df_ = df[(df.epoch==500)].groupby(['percentage', 'type'], as_index=False)['accuracy_semimanual'].agg({'acc_std':'std', 'acc_mean':'mean'})

    df_n = df[(df.epoch==500)].groupby(['percentage', 'type'], as_index=False).agg(lambda x: np.mean(x))

    logger.info(f'Available model types: {df_n.type.unique()}')

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

    logger.info(f"mean cellpose accuracies: {df_[selection]['acc_mean']}")

    ax1.errorbar(df_n[selection].cell_number, df_[selection]['acc_mean'],
                yerr=df_[selection]['acc_std'], color=l.get_color(), ls='-',
                label='cellpose')
    ax1.set_xlabel('Cell number')
    ax1.set_ylabel('accuracy [a.u.]')
    ax1.legend()
    ax1.grid()

    output_dir.mkdir(parents=True, exist_ok=True)

    f.savefig(output_dir / 'fig3c.png')
    f.savefig(output_dir / 'fig3c.svg')

    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("training_data_stardist", type=Path)
    parser.add_argument("--cellpose_accuracies", type=Path, nargs='+')
    parser.add_argument("--stardist_accuracies", type=Path, nargs='+')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_data_abundance_vs_accuracy(
        args.output_dir,
        args.training_data_stardist,
        args.cellpose_accuracies,
        args.stardist_accuracies,
    )

    return

if __name__ == "__main__":
    main()


