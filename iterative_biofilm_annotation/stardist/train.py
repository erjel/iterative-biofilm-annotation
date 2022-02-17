import os
import numpy as np

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D, StarDistData3D

from skimage.segmentation import relabel_sequential

from pathlib import Path
from argparse import ArgumentParser

from typing import Tuple


def random_fliprot(img, mask, axis=None): 
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y


def train_stardist(model_folder: Path, dataset_folder: Path, train_patch_size: Tuple[int]):

    modelname = model_folder.name
    basedir = model_folder.parent


    n_rays = 192
    del_empty_patches = False
    percentage = 100

    X_trn_paths = sorted((dataset_folder / 'train' / 'images').glob('*.tif'))
    X_vld_paths = sorted((dataset_folder / 'valid' / 'images').glob('*.tif'))

    for p in X_trn_paths:
        print(p)
        
    for p in X_vld_paths:
        print(p)

    X_trn = [imread(p) for p in tqdm(X_trn_paths)]
    X_vld = [imread(p) for p in tqdm(X_vld_paths)]


    Y_trn_paths = sorted((dataset_folder / 'train' / 'masks').glob('*.tif'))
    Y_vld_paths = sorted((dataset_folder / 'valid' / 'masks').glob('*.tif'))

    for p in Y_trn_paths:
        print(p)
        
    for p in Y_vld_paths:
        print(p)

    Y_trn = [imread(p) for p in tqdm(Y_trn_paths)]
    Y_vld = [imread(p) for p in tqdm(Y_vld_paths)]

    X_trn[2] = X_trn[2][1:] # Strange ...

    for y, x in zip(Y_trn, X_trn):
        print(y.shape, x.shape)

    axis_norm = (0, 1, 2)
    X_trn= [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X_trn)]
    X_vld= [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X_vld)]

    n_channel = 1

    extents = calculate_extents(Y_trn[2])
    anisotropy = tuple(np.max(extents) / extents)

    use_gpu = gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D (
        rays=rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size=train_patch_size,
        train_batch_size=1,
    )
    vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.75, total_memory=10000)

    model = StarDist3D(conf,
                       name=modelname,
                       basedir=basedir)

    fov = np.array(model._axes_tile_overlap('ZYX'))

    median_size = calculate_extents(Y_trn, np.median)

    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    model.train(X_trn, Y_trn,
                validation_data=(X_vld, Y_vld),
                epochs=1000,
                augmenter=augmenter)

    model.optimize_thresholds(X_vld, Y_vld)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('model_folder', type=Path, help="Full model folder (including model name)")
    parser.add_argument('dataset_folder', type=Path, help="Dataset folder")
    parser.add_argument('--patch_size', type=str, default='64x128x128')

    return parser, parser.parse_args()

def main():
    parser, args = parse_args()

    patch_size = tuple(int(v) for v in args.patch_size.split('x'))

    train_stardist(
        args.model_folder,
        args.dataset_folder,
        patch_size        
    )


if __name__ == "__main__":
    main()