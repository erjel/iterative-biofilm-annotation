from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
from skimage.segmentation import relabel_sequential
from skimage.io import imread
from stardist import fill_label_holes
from tqdm import tqdm

import os
import sys
sys.path.append(str(Path(os.getcwd())))
from iterative_biofilm_annotation.stardist.data import (
    sliceToShape, writeDatasetFolders, correctFullStackOffset, ImageInterpolation
)


def create_partial_dataset(
    dataset_path: Path,
    patchsize: str,
    label_files: List[Path],
    image_files: List[Path],
) -> None:

    for files in [label_files, image_files]:
        for f in files:
            assert f.is_file()

    assert len(label_files) == len(image_files)
    
    X = []
    Y = []

    for label_file, image_file in zip(sorted(label_files), sorted(image_files)):

        label = imread(label_file)
        img = imread(image_file)[1:]

        assert img.shape == label.shape, \
            f'{img.shape} == {label.shape}'
        
        label, img = correctFullStackOffset(label, img, verbose=True)

        X.append(img)
        Y.append(label)


    Y = [relabel_sequential(y)[0] for y in tqdm(Y)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]
    
    X, Y = sliceToShape([X, Y], output_shape=tuple(int(s) for s in patchsize.split('x')))

    # split randomly
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_test = max(1, int(round(0.10 * len(ind))))
    ind_train, ind_test = ind[:-n_test], ind[-n_test:]
    X_test, Y_test = [X[i] for i in ind_test], [Y[i] for i in ind_test]
    X_trn,  Y_trn  = [X[i] for i in ind_train],[Y[i] for i in ind_train]

    ind = rng.permutation(len(X_trn))
    n_val = max(1, int(round(0.10 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val   = [X_trn[i] for i in ind_val],  [Y_trn[i] for i in ind_val]
    X_trn,  Y_trn  = [X_trn[i] for i in ind_train],[Y_trn[i] for i in ind_train]

    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))
    print('- test:           %3d' % len(X_test))

    datasets = [X_val, X_trn, X_test, Y_val, Y_trn, Y_test]
    dataPurposes = ['valid', 'train', 'test']*2
    dataTypes = ['images']*3 + ['masks']*3

    for dataset, dataPurpose, dataType in zip(datasets, dataPurposes, dataTypes):
        writeDatasetFolders(dataset, dataset_path.name, dataPurpose, dataType)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset_path', type=Path)
    parser.add_argument('patchsize', type=str)
    parser.add_argument('--label_files', type=Path, nargs='+')
    parser.add_argument('--image_files', type=Path, nargs='+')
    args = parser.parse_args()


    create_partial_dataset(
        args.dataset_path,
        args.patchsize,
        args.label_files,
        args.image_files,
    )
