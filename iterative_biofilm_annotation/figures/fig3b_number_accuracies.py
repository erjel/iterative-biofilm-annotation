from argparse import ArgumentParser
from tifffile import imread
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt


def getResultsAndFeatures(accuracy_file, skip):
    results = np.genfromtxt(accuracy_file, delimiter=',', skip_header=skip)

    with open(accuracy_file, 'r') as f:
        features = f.readline()[2:-1].split(',')            

    return results, features

def plot_number_accuracies() -> None:

results_lists = [ np.genfromtxt(accuracy_file, delimiter=',', skip_header=1) for accuracy_file in accuracy_files]

with open(accuracy_files[0], 'r') as f:
    features = f.readline()[2:-1].split(',')



f, (ax1) = plt.subplots(1, 1, figsize=(7, 5))

ls_list = ['-', '--', '-.', ':']
labels = [' Stardist', ' Improved Hartmann et al.', ' Hartmann et al.', ' Cellpose']

for idx in [2, 3, 4]:    
    results = results_lists[0]
    ls = ls_list[0]
    label = labels[0]
    
    l, = ax1.plot(results[:, 1], results[:, idx], label=features[idx] +  label,linestyle=ls, linewidth=2)
    
    for results, ls, label in zip(results_lists[1:], ls_list[1:], labels[1:]):

        l, = ax1.plot(results[:, 1], results[:, idx], label=features[idx] + label, color=l.get_color(), linestyle=ls, linewidth=2)
        
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax1.grid()
ax1.set_xlabel('IoU threshold (a.u.)')
ax1.set_ylabel('Counts');

output_folder = Path('outputs')

if not output_folder.is_dir():
    os.makedirs(str(output_folder))

plt.savefig(str(output_folder / 'Number_accuracy.svg'), bbox_inches='tight')

f, axes = plt.subplots(1, 3, figsize=(14, 3.5))


ls_list = ['-', '--', '-.', ':']
labels = ['Stardist', 'Improved Hartmann et al.', 'Hartmann et al.', 'Cellpose']

for idx,ax1 in zip([2, 3, 4], axes.flat):    
   
    for results, ls, label in zip(results_lists, ls_list, labels):

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
    parser.add_argument(output_folder, type=Path)
    parser.add_argument('--labels', type=str, nargs='+')
    parser,

    return parser.parse_args()

def main():
    args = parse_args()

    accuracy_files = [
        # TODO(erjel): Use mean value instead of single calculation for stardist & cellpose 
        r"Y:\Eric\2021_Iterative_Biofilm_Annotation\data\stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep5\accuracy_full_semimanual-raw_verbose.csv", # stardist
        r"Y:\Eric\2021_Iterative_Biofilm_Annotation\data_BiofilmQ\full_stacks_huy\accuracies\data_seeded_watershed\Pos1_ch1_frame000001_Nz300.csv", # Improved Hartmann et al
        r"Y:\Eric\2021_Iterative_Biofilm_Annotation\data_BiofilmQ\full_stacks_huy\accuracies\data_hartmann_et_al\Pos1_ch1_frame000001_Nz300.csv", # Hartmann et al
        r"Y:\Eric\2021_Iterative_Biofilm_Annotation\data\horovod_cellpose_patches-semimanual-raw-64x128x128_prc100_bs8_lr0.00625_wd0.00001_mt0.7_sge_rep1\accuracy_full_semimanual-raw_verbose.csv" # cellpose
    ]

    plot_number_accuracies(

    )

    return

if __name__ == '__main__':
    main()