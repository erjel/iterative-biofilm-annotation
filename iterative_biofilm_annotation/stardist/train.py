import os
from random import seed
import numpy as np

from glob import glob
import hashlib

from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
import yaml

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D, StarDistData3D
import tensorflow as tf

from skimage.segmentation import relabel_sequential

from data import readDataset
import argparse


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

def train(modelname, basedir, dataset_name,n_rays, train_patch_size, del_empty_patches, percentage):

    X, Y = readDataset(dataset_name)
    
    X['test'] = []
    Y['test'] = []
   
    if del_empty_patches:
        for s in X.keys():
            sum_Y = [np.sum(y) for y in Y[s]]
            X[s] = [X[s][i] for i in range(len(X[s])) if sum_Y[i] > 0]
            Y[s] = [Y[s][i] for i in range(len(Y[s])) if sum_Y[i] > 0]
            
    axis_norm = (0, 1, 2)
    X['train'] = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X['train'])]
    X['valid'] = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X['valid'])]

    seed = int(hashlib.sha1(modelname.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    print(f'Use random seed: {seed}')
    tf.keras.utils.set_random_seed(seed)
    assert(len(X['train']) > 1)
    print('Number of training patches: ', len(X['train']))
    rng = np.random.RandomState()
    ind = rng.permutation(len(X['train']))
    n_val = max(1, int(round(percentage / 100 * len(ind))))
    print('Number of training patches: ', n_val)

    X['train'] = [X['train'][i] for i in ind[:n_val]]
    Y['train'] = [Y['train'][i] for i in ind[:n_val]]

    labels_in_train_samples = [np.unique(y) for y in Y['train']]
    num_cells_in_train_samples = [len(labels[labels != 0]) for labels in labels_in_train_samples]

    labels_in_valid_samples = [np.unique(y) for y in Y['valid']]
    num_cells_in_valid_samples = [len(labels[labels != 0]) for labels in labels_in_valid_samples]

    training_infos = {
        'num_cells_in_training_samples': num_cells_in_train_samples,
        'num_cells_in_valid_samples': num_cells_in_valid_samples,
        'seed': seed,
    }

    training_info_file = Path(basedir) / modelname / 'training_infos.yaml'
    training_info_file.parent.mkdir(parents=True, exist_ok=True)

    with open(training_info_file, 'w') as f:
        f.write(yaml.dump(training_infos))
   
    print(Config3D.__doc__)
    n_channel = 1

    extents = calculate_extents(Y['train'][0])
    anisotropy = tuple(np.max(extents) / extents)

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
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
        train_batch_size=2,
    )
    print(conf)
    vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8)

    model = StarDist3D(conf,
                       name=modelname,
                       basedir=basedir)

    fov = np.array(model._axes_tile_overlap('ZYX'))

    median_size = calculate_extents(Y['train'], np.median)

    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    model.train(X['train'], Y['train'],
                validation_data=(X['valid'], Y['valid']),
                epochs=400,
                augmenter=augmenter)

    model.optimize_thresholds(X['valid'], Y['valid'])

def parse_args():
    parser = argparse.ArgumentParser(description='Conduct stardist training with the given input parameter')
    parser.add_argument('modelname', metavar='MODELNAME', type=str, help="Name for saving the model")
    parser.add_argument('basedir', metavar='BASEDIR', type=str, help="directory for models")
    parser.add_argument('dataset_name', metavar='DATASET', type=str, help="Dataset name for figure title")
    parser.add_argument('n_rays', metavar='NUMRAYS', type=int)
    parser.add_argument('train_patch_size', metavar='PATCHSIZE', type=str)
    parser.add_argument('del_empty_patches', metavar='DELEMPTY', type=str)
    parser.add_argument('--percentage', metavar='PERCENTAGE', type=float, default=100)
    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    
    train_patch_size=tuple(int(s) for s in args.train_patch_size.split('x'))
    del_empty_patches = bool(args.del_empty_patches)

    train(args.modelname, args.basedir, args.dataset_name, args.n_rays, train_patch_size, del_empty_patches, args.percentage)

if __name__ == '__main__':
    main()
