from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def plot_accuracy_comparison():
    
acc_files_horovod = sorted((data_folder / 'data').glob(r'horovod*prc100*\accuracy_full_semimanual-raw.csv'))

accuracy_files_ = [(data_folder / 'data_BiofilmQ' / r'full_stacks_huy\accuracies\data_seeded_watershed\Pos1_ch1_frame000001_Nz300.csv')]
accuracy_files_hartmann_et_al = [(data_folder / 'data_BiofilmQ' / r'full_stacks_huy\accuracies\data_hartmann_et_al\Pos1_ch1_frame000001_Nz300.csv')]

#TODO(erjel): Read from config
stardist_models_raw = \
[r"stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep1",
r"stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep2",
r"stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep3",
r"stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep4",
r"stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep5"]

stardist_models_raw = [str(data_folder / 'data' / name) for name in stardist_models_raw]

acc_files_stardist = [Path(m) / 'accuracy_full_semimanual-raw.csv' for m in stardist_models_raw]

accuracy_files = accuracy_files_ + accuracy_files_hartmann_et_al + acc_files_horovod+ acc_files_stardist
output_folder = Path('outputs')
if not output_folder.is_dir():
    os.makedirs(output_folder)
figure_filename = str(output_folder / 'segmentation_combarison.svg')

print(figure_filename)
print(accuracy_files)

modelnames = [Path(f).parent.name for f in accuracy_files]


print('\n')
data = []
for i, filename in enumerate(accuracy_files):
    print(filename)
    if i > 1:
        data.append(np.genfromtxt(filename, delimiter=' '))
    else:
        data.append(np.genfromtxt(str(filename), delimiter=',', skip_header=1))

modelnames_clean = [m.split('_rep')[0] for m in modelnames]
model_type_names, model_types = np.unique(modelnames_clean, return_inverse=True)
model_type_names = ['Improved Hartmann et al.', 'Hartmann et al.', 'Cellpose', 'Stardist']

f, ax = plt.subplots(1)

for model_type in range(max(model_types)+1):
    models_of_type = np.where(model_types == model_type)[0]
    data_ = [data[i] for i in models_of_type]
    print(len(data_))
    data_ = np.asarray(data_)
    print(data_.ndim, data_.shape)
    mean = np.mean(data_, axis=0)
    std = np.std(data_, axis=0)
    
    if model_type > 1:
    
        p, = ax.plot(mean[0], mean[1], label=model_type_names[model_type])

        ax.fill_between(mean[0], mean[1] - std[1], mean[1] + std[1],
                    color=p.get_color(), alpha=0.2)
    else:
        print(sum(mean))
        ax.plot(np.asarray(data[model_type][:, 1]), data[model_type][:,7], label=model_type_names[model_type])

x_label = 'Intersection over union [a.u.]'
y_label = 'Average precision [a.u.]'
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)

ax.grid()
ax.legend()
plt.savefig(figure_filename, bbox_inches='tight')
plt.savefig(figure_filename.split('.')[0] + '.png', bbox_inches='tight')
plt.show()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--stardist_accuracies', type=Path, nargs='+')
    parser.add_argument('--cellpose_accuracies', type=Path, nargs='+')
    parser.add_argument('--biofilmq_improved_accuracies', type=Path)
    parser.add_argument('--biofilmq_accuracies', type=Path)
    parser.add_argument('--stardist_improved_accuracies', type=Path, default=[])

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    plot_accuracy_comparison(

    )
    
if __name__ == "__main__":
    main()