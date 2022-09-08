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
    stardist_improved_accuracies: Optional[List[Path]] = None,
    unet_accuracies: Optional[List[Path]] = None,
    ) -> None:

    accuracy_files = [
        stardist_accuracies,
        cellpose_accuracies,
        biofilmq_accuracies,
        biofilmq_improved_accuracies,
        stardist_improved_accuracies,
        unet_accuracies,
    ]
    
    f, ax = plt.subplots(1)

    for label, accuracy_list in zip(model_type_names, accuracy_files):
        data = [np.genfromtxt(str(filename), delimiter=',', skip_header=1)[:, [1,7]] for filename in accuracy_list]
        data = np.asarray(data)
        logger.info(f'{data.shape}')
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        logger.info(mean)
        
        logger.info(f'length for "{label}": {len(data)}')

        p, = ax.plot(mean[:, 0], mean[:, 1], label=label)

        ax.fill_between(mean[:, 0], mean[:, 1] - std[:, 1], mean[:, 1] + std[:, 1],
            color=p.get_color(), alpha=0.2)

    x_label = 'Intersection over union [a.u.]'
    y_label = 'Average precision [a.u.]'
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid()
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_filename = output_dir / 'segmentation_comparison'

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
    parser.add_argument('--unet_accuracies', type=Path, nargs='+')

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
        args.unet_accuracies,
    )
    
if __name__ == "__main__":
    main()