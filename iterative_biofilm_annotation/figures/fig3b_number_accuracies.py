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
        biofilmq_accuracies: List[Path]
    ) -> None:

    # TODO(erjel): Dirty hack which uses only N=1 sample size for the curves
    accuracy_files = [
        stardist_accuracies[0],
        biofilmq_improved_accuracies[0],
        biofilmq_accuracies[0],
        cellpose_accuracies[0]
    ]

    results_lists = [np.genfromtxt(accuracy_file, delimiter=',', skip_header=1) for accuracy_file in accuracy_files]

    with open(accuracy_files[0], 'r') as f:
        features = f.readline()[2:-1].split(',')

    f, (ax1) = plt.subplots(1, 1, figsize=(7, 5))


    for idx in [2, 3, 4]:    
        results = results_lists[0]
        ls = plotstyle[0]
        label = labels[0]
        
        l, = ax1.plot(results[:, 1], results[:, idx], label=features[idx] +  label,linestyle=ls, linewidth=2)
        
        for results, ls, label in zip(results_lists[1:], plotstyle[1:], labels[1:]):

            l, = ax1.plot(results[:, 1], results[:, idx], label=features[idx] + label, color=l.get_color(), linestyle=ls, linewidth=2)
            
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax1.grid()
    ax1.set_xlabel('IoU threshold (a.u.)')
    ax1.set_ylabel('Counts');

    if not output_folder.is_dir():
        os.makedirs(str(output_folder))

    plt.savefig(str(output_folder / 'Number_accuracy.svg'), bbox_inches='tight')

    f, axes = plt.subplots(1, 3, figsize=(14, 3.5))



    for idx,ax1 in zip([2, 3, 4], axes.flat):    
    
        for results, ls, label in zip(results_lists, plotstyle, labels):

            l, = ax1.plot(results[:, 1], results[:, idx], label= label, linestyle=ls, linewidth=2)
            
        ax1.grid()
        ax1.set_xlabel('IoU threshold (a.u.)')
        ax1.set_ylabel('Counts');
        ax1.set_title(features[idx])

    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

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
        args.biofilmq_accuracies
    )

    return

if __name__ == '__main__':
    main()