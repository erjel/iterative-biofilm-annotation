from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

# imports for data preparation
from tifffile import TiffFile
from csbdeep.utils import download_and_extract_zip_file, plot_some, _raise
from csbdeep.data import create_patches, RawData
from csbdeep.utils import Path

import os

import pandas as pd

from argparse import ArgumentParser

def readTiffPage(path, i):
    with TiffFile(path) as tif:
        return tif.pages[i].asarray()

def toolbox2RawData(slice_df, axes='CYX', lowChannel='ch2', gtChannel='ch1', test_split=None):
    from csbdeep.utils import Path, consume, axes_check_and_normalize
    from posixpath import join


    z_heights = slice_df.std_argmax + 1 # overview plane is not account during calculation

    source_paths = [str(Path(p)) for p in slice_df.Path]
    target_paths = [str(Path(p)).replace(lowChannel, gtChannel) for p in slice_df.Path]
    
    if test_split is not None:
        idcs = np.arange(len(source_paths))
        np.random.shuffle(idcs)
        split_idx = int(round(len(source_paths)*test_split))
        idcs_train = idcs[split_idx:]
        idcs_test = idcs[:split_idx]

        source_paths_train =  [source_paths[i] for i in idcs_train] 
        target_paths_train = [target_paths[i] for i in idcs_train]
        idx_train = [z_heights[i] for i in idcs_train]
        
        source_paths_test =  [source_paths[i] for i in idcs_test]
        target_paths_test = [target_paths[i] for i in idcs_test]
        idx_test = [z_heights[i] for i in idcs_test]
        
    else:
        source_paths_train, target_paths_train, idx_train = source_paths, target_paths, idx_train
        source_paths_test, target_paths_test, idx_test = [], [], []
    

    len(source_paths_train) > 0 or _raise(FileNotFoundError("'target_dir' doesn't exist or didn't find any images in it."))

    consume ((
       os.path.exists(s) or _raise(FileNotFoundError(s))
       for s in source_paths_train
    ))

    axes = axes_check_and_normalize(axes)
    


    xy_name_pairs_train = list(zip(source_paths_train, target_paths_train, idx_train))
    xy_name_pairs_test = list(zip(source_paths_test, target_paths_test, idx_test))

    n_images_train = len(xy_name_pairs_train)
    n_images_test = len(xy_name_pairs_test)
    
    print(f'Train images: {n_images_train}', '\n'
          f'Test images:  {n_images_test}', '\n'
          f'Total images: {len(source_paths)}')

    description_train = "target='{o}', sources={s}, axes='{a}'".format(
        s=list(source_paths_train),
        o=list(target_paths_train),
        a=axes)
    

    def _gen():
        for fx, fy, i in xy_name_pairs_train:
            x, y = readTiffPage(str(fx), i) , readTiffPage(str(fy), i)
            # print(x.shape, y.shape)
            x.shape == y.shape or _raise(ValueError())
            len(axes) >= x.ndim or _raise(ValueError())
            yield x, y, axes[-x.ndim:], None

    return RawData(_gen, n_images_train, description_train), xy_name_pairs_test


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('output_npz', type=str)
    parser.add_argument('cover_slip_slice_csv', type=str)
    parser.add_argument('--seed', type=int, default=7848)

    return parser, parser.parse_args()

def main():
    parser, args = parse_args()

    slice_csv = Path(args.cover_slip_slice_csv)
    np.random.seed(args.seed)
    
    slice_df = pd.read_csv(slice_csv)
    slice_df.std_argmax  = slice_df.std_argmax

    raw_data_train, test_data = toolbox2RawData (
        slice_df,
        test_split  = 0.1
    )

    
    X, Y, XY_axes = create_patches (
        raw_data            = raw_data_train,
        patch_size          = (64,64),
        n_patches_per_image = 100,
        save_file           = args.output_npz,
    )

    
    return

if __name__ == '__main__':
    main()