from argparse import ArgumentParser
import logging
from typing import List
import sys
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def getResultsAndFeatures(accuracy_file, skip):
    results = np.genfromtxt(accuracy_file, delimiter=',', skip_header=skip)

    with open(accuracy_file, 'r') as f:
        features = f.readline()[2:-1].split(',')            

    return results, features

def plot_number_accuracies(
        output_folder: Path,
        labels: List[str],
        plotstyle: List[str],
        cellpose_accuracies: List[Path],
        stardist_accuracies: List[Path],
        biofilmq_improved_accuracies: List[Path],
        biofilmq_accuracies: List[Path],
        stardist_improved_accuracies: List[Path],
        unet_accuracies: List[Path],

    ) -> None:

    accuracy_files_list = [
        cellpose_accuracies,
        stardist_accuracies,
        biofilmq_improved_accuracies,
        biofilmq_accuracies,
        stardist_improved_accuracies,
        unet_accuracies,
    ]

    # Get feature names from csv
    with open(accuracy_files_list[0][0], 'r') as f:
        features = f.readline()[2:-1].split(',')

    f, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    for idx, ax in zip([2, 3, 4], axes.flat):    
    
        for accuracy_files, ls, label in zip(accuracy_files_list, plotstyle, labels):
            
            data = [np.genfromtxt(accuracy_file, delimiter=',', skip_header=1) for accuracy_file in accuracy_files]
            data = np.asarray(data)
            logger.info(f'data shape: {data.shape}')
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            logger.info(f'label: {label}')
            logger.info(mean)

            p, = ax.plot(mean[:, 1], mean[:, idx], label= label, linestyle=ls, linewidth=2)
            
            ax.fill_between(mean[:, 1], mean[:, idx] - std[:, idx], mean[:, idx] + std[:, idx],
                color=p.get_color(), alpha=0.2)

        ax.grid()
        ax.set_xlabel('IoU threshold (a.u.)')
        ax.set_ylabel('Counts');
        ax.set_title(features[idx])

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    output_folder.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(output_folder / 'Number_accuracy.svg'), bbox_inches='tight')
    plt.savefig(str(output_folder / 'Number_accuracy.png'), bbox_inches='tight')

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("output_folder", type=Path)
    parser.add_argument('--labels', type=str, nargs='+')
    parser.add_argument('--plotstyle', type=str, nargs='+')
    parser.add_argument('--cellpose_accuracies', type=Path, nargs='+')
    parser.add_argument('--stardist_accuracies', type=Path, nargs='+')
    parser.add_argument('--biofilmq_improved_accuracies', type=Path, nargs='+')
    parser.add_argument('--biofilmq_accuracies', type=Path, nargs='+')
    parser.add_argument('--stardist_improved_accuracies', type=Path, nargs='+')
    parser.add_argument('--unet_accuracies', type=Path, nargs='+')

    return parser.parse_args()

def main():
    args = parse_args()

    plot_number_accuracies(
        args.output_folder,
        args.labels,
        args.plotstyle,
        args.cellpose_accuracies,
        args.stardist_accuracies,
        args.biofilmq_improved_accuracies,
        args.biofilmq_accuracies,
        args.stardist_improved_accuracies,
        args.unet_accuracies,
    )

    return

if __name__ == '__main__':
    main()