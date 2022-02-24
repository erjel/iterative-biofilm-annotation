from argparse import ArgumentParser, Namespace
from genericpath import exists
from multiprocessing import dummy
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_accuracy_comparison(
    output_dir: Path,
    model_type_names: List[str],
    stardist_accuracies: List[Path],
    cellpose_accuracies: List[Path],
    biofilmq_improved_accuracies: List[Path],
    biofilmq_accuracies: List[Path],
    stardist_improved_accuracies: Optional[List[Path]] = None
    ) -> None:

    accuracy_files = \
        stardist_accuracies + \
        cellpose_accuracies + \
        biofilmq_accuracies + \
        biofilmq_improved_accuracies + \
        stardist_improved_accuracies
    
    [logger.info(a) for a in accuracy_files]

    modelnames = [Path(f).parent.name for f in accuracy_files]

    logger.info(modelnames)


    data = []
    for i, filename in enumerate(accuracy_files):
        logger.info(filename)
        data_ = np.genfromtxt(str(filename), delimiter=',', skip_header=1)[:, [1, 7]]
        data.append(data_)

    modelnames_clean = [m.split('_rep')[0] for m in modelnames]
    dummy, model_types = np.unique(modelnames_clean, return_inverse=True)

    f, ax = plt.subplots(1)

    for model_type in range(max(model_types)+1):
        models_of_type = np.where(model_types == model_type)[0]
        data_ = [data[i] for i in models_of_type]
        logger.info(len(data_))
        data_ = np.asarray(data_)
        logger.info(f'{data_.shape}')
        mean = np.mean(data_, axis=0)
        std = np.std(data_, axis=0)
        logger.info(mean)
        
        logger.info(f'length for {model_type_names[model_type]}: {len(models_of_type)}')

        p, = ax.plot(mean[:, 0], mean[:, 1], label=model_type_names[model_type])

        ax.fill_between(mean[:, 0], mean[:, 1] - std[:, 1], mean[:, 1] + std[:, 1],
            color=p.get_color(), alpha=0.2)

    x_label = 'Intersection over union [a.u.]'
    y_label = 'Average precision [a.u.]'
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid()
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_filename = output_dir / 'segmentation_comparison.svg'

    plt.savefig(f"{str(figure_filename)}.svg", bbox_inches='tight')
    plt.savefig(f"{str(figure_filename)}.png", bbox_inches='tight')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--labels', type=str, nargs='+')
    parser.add_argument('--stardist_accuracies', type=Path, nargs='+')
    parser.add_argument('--cellpose_accuracies', type=Path, nargs='+')
    parser.add_argument('--biofilmq_improved_accuracies', type=Path, nargs='+')
    parser.add_argument('--biofilmq_accuracies', type=Path, nargs='+')
    parser.add_argument('--stardist_improved_accuracies', type=Path, nargs='+', default=[])

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    plot_accuracy_comparison(
        args.output_dir,
        args.labels,
        args.stardist_accuracies,
        args.cellpose_accuracies,
        args.biofilmq_improved_accuracies,
        args.biofilmq_accuracies,
        args.stardist_improved_accuracies,
    )
    
if __name__ == "__main__":
    main()