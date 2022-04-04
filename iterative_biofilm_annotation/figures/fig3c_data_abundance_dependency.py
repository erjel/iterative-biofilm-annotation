from argparse import ArgumentParser, Namespace
from functools import partial
import logging
from pathlib import Path
import re
import sys
from typing import List
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    stardist_merge_accuracies: List[Path],
    ) -> None:

    accuracy_files_dict = {
        'stardist': stardist_accuracies,
        'cellpose': cellpose_accuracies,
        'stardist_merge': stardist_merge_accuracies,
    }


    # Prepare cell number counts
    Y = list(map(imread, training_data_stardist.glob('train/masks/*.tif')))

    logger.info(f'Training data: {training_data_stardist}')
    logger.info(f"Number of training images: {len(Y)}")

    # Delete all empty images to reconstruct the correct training image order
    # TODO(erjel): Check this!
    Y = [y for y in Y if y.max() > 0]

    logger.info(f'Number of non-empty training images {len(Y)}')

    im_labels = [np.unique(y) for y in Y]
    im_labels = [l[l!=0] for l in im_labels]

    N_cells = [len(l) for l in im_labels]

    def get_accuracy_vs_percent_cellnumber(
        model_type: str,
        accuracy_files: List[Path],
        N_cells: List[int]
        ) -> np.ndarray:

        # Read metadata from accuracy paths
        p = '.*True_(?P<percentage>[\d\.]+)prc_rep(?P<replicate>\d+)'
        pattern = re.compile(p)

        path_metadata = [pattern.match(str(f)).groupdict() for f in accuracy_files]

        df = pd.DataFrame(path_metadata)
        df['path'] = accuracy_files
        df = df.astype({'percentage': float, 'replicate':int})
        df['seed'] = df.replicate.astype(int) #TODO(erjel): USE THIS DURING TRAINING!


        # Estimate the cell number
        for index, row in df.iterrows():
            accuracy_path = Path(row.path)
            training_info_path = Path('models') / accuracy_path.parent.name / 'training_infos.yaml'

            if training_info_path.is_file():
                with open(training_info_path, 'r') as f:
                    training_info = yaml.load(f, Loader=yaml.BaseLoader)

                df.loc[index, 'cell_number'] = training_info['num_cells_in_training_samples']

            else:
                # Dirty hack:
                seed = row.seed
                rng = np.random.RandomState(seed)
                ind = rng.permutation(len(Y))
                n_val = max(1, int(round(float(row.percentage) / 100 * len(ind))))

                df.loc[index, 'cell_number'] = np.sum([N_cells[i] for i in ind[:n_val]])

                # Better solution (given that the labels per image (im_labels) are known)
                #seed = row.seed
                #rng = np.random.RandomState(seed)
                #ind = rng.permutation(len(Y))
                #n_val = max(1, int(round(float(row.percentage) / 100 * len(ind))))
                #
                #df.loc[index, 'cell_number'] = len(np.unique(np.concatenate([im_labels[i] for i in ind[:n_val]])))

        # Read accuracies
        accuracy_manual = []
        for path in df.path:
            data = np.genfromtxt(path, delimiter=' ')
            accuracy_manual.append(data[1][np.where(data[0]==0.5)[0]][0])

        df['accuracy'] = accuracy_manual

        df = df.groupby(['percentage'], as_index=False).agg({'accuracy': ['mean', 'std'], 'cell_number': ['mean', 'std']})

        return df


    get_accuracy_vs_percent = partial(get_accuracy_vs_percent_cellnumber, N_cells = N_cells)

    f, ax = plt.subplots(1,1,)
    for model_type, accuracy_files in accuracy_files_dict.items():
        data = get_accuracy_vs_percent(model_type, accuracy_files)
        print(data)

        #ax.errorbar(
        #    x = data.cell_number['mean'],
        #    y = data.accuracy['mean'],
        #    yerr = data.accuracy['std'],
        #    xerr = data.cell_number['std'],
        #    label = model_type
        #)

        l, = ax.plot(data.cell_number['mean'], data.accuracy['mean'], label = model_type )

        ax.fill_between(
            x = data.cell_number['mean'],
            y1 = data.accuracy['mean'] - data.accuracy['std'] ,
            y2 = data.accuracy['mean'] + data.accuracy['std'] ,
            alpha= 0.2,
            color = l.get_color()
        )

    ax.set_xlabel('Cell number')
    ax.set_ylabel('accuracy [a.u.]')
    ax.legend()
    ax.grid()

    output_dir.mkdir(parents=True, exist_ok=True)
    
    [f.savefig(output_dir / f'data_abundance_dependency.{ext}') for ext in ['png', 'svg']]

    return
    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("training_data_stardist", type=Path)
    parser.add_argument("--cellpose_accuracies", type=Path, nargs='+')
    parser.add_argument("--stardist_accuracies", type=Path, nargs='+')
    parser.add_argument("--stardist_merge_accuracies", type=Path, nargs='+')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_data_abundance_vs_accuracy(
        args.output_dir,
        args.training_data_stardist,
        args.cellpose_accuracies,
        args.stardist_accuracies,
        args.stardist_merge_accuracies,
    )

    return

if __name__ == "__main__":
    main()


