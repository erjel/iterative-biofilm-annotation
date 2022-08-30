from glob import glob

import os
from tifffile import imread, imsave
from skimage.segmentation import relabel_sequential
from stardist import fill_label_holes
from tqdm import tqdm

from csbdeep.utils import normalize
import numpy as np
from pathlib import Path
import argparse

import sys
sys.path.append(str(Path(os.getcwd())))
from iterative_biofilm_annotation.stardist.data import sliceToShape, writeDatasetFolders, correctFullStackOffset, ImageInterpolation

def main(annotation_path, raw_path, dataset_path, filetype, patch_size, exclude_file, dz):

    annotation_path = Path(annotation_path)
    raw_path = Path(raw_path)

    print(annotation_path, annotation_path.is_dir())

    label_paths = sorted(annotation_path.glob('**/*0?/*0L.tif'))
    label_paths = [p for  p in label_paths if not p == Path(exclude_file)]

    dataset_path = Path(dataset_path)

    if filetype == 'huy':
        ch_in = 'ch1'
        ch_out = 'ch2'
    elif filetype == 'raw':
        ch_in = 'ch2'
        ch_out = 'ch1'
    else:
        error()
        
    X = []
    Y = []

    for label_path in label_paths:

        label = imread(str(label_path))
        label_path = label_path.relative_to(annotation_path)
        raw_dir = raw_path / Path(*label_path.parts[:-2], label_path.name.replace('L.tif','.tif').replace(ch_in, ch_out))
        print(raw_dir)
        raw_file = imread(str(raw_dir))
        
        print(raw_file.shape, label.shape)
        
        label, raw_file = correctFullStackOffset(label, raw_file, verbose=True)

        X.append(raw_file)
        Y.append(label)


    Y = [relabel_sequential(y)[0] for y in tqdm(Y)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]
    
    if not dz is None:
        factor = int(dz/100)
        X = [ImageInterpolation(x[::factor], factor, x.shape) for x in X]

    X, Y = sliceToShape([X, Y], output_shape=tuple(int(s) for s in patch_size.split('x')))

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
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_path', metavar='LABELPATH', type=str)
    parser.add_argument('raw_path', metavar='RAWPATH', type=str)
    parser.add_argument('dataset_path', metavar='DATASETPATH', type=str)
    parser.add_argument('filetype', metavar='FILETYPE', type=str)
    parser.add_argument('patchsize', metavar='PATCHSIZE', type=str)
    parser.add_argument('exclude', metavar='EXCLUDE', type=str)
    parser.add_argument('--intp', metavar='dz', type=float, help="stepping to delete from full stack with dz=100")
    args = parser.parse_args()


    main(
        args.annotation_path,
        args.raw_path,
        args.dataset_path,
        args.filetype,
        args.patchsize,
        args.exclude,
        args.intp
    )
