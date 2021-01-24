#!/usr/bin/env python
from pathlib import Path
import os
from tqdm import tqdm
from tifffile import imsave, imread
from stardist.models.model3d import StarDist3D
import sys
from csbdeep.utils import normalize
import numpy as np
import argparse

sys.path.append(str(Path(os.getcwd())))

from stardist_mpcdf.data import readDataset

def main(model_path, dataset_path, output_path, use_overlap):

    stardist_mpcdf_home = Path(os.getcwd()).parent

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    dataset_name = dataset_path.stem
    modelname = model_path.name
    model_basedir = model_path.parent

    print(dataset_path)
    if dataset_path.is_dir():

        X = readDataset(dataset_name)[0]
        X = X['test']

        input_dir = dataset_path / 'test' / 'images'
        output_dir = output_path / 'test' / 'images'
        filelist_x = sorted(p for p in input_dir.glob('*.tif'))


    elif dataset_path.is_file():
        X = [imread(str(dataset_path))]
        X = [x[1:] if np.argmax(np.sum(x, axis=(1, 2)), axis=0) == 0 else x for x in X]
        output_dir = Path('interference') / dataset_name / modelname
        if use_overlap:
            filelist_x = [Path(str(dataset_path).replace('.tif', 'O.tif'))]
        else:
            filelist_x = [Path(str(dataset_path).replace('.tif', 'P.tif'))]


    axis_norm = (0, 1, 2)
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]

    print(dataset_name)

    print('Dataset length: ', len(X))

    print('Load model "{}"'.format(modelname))
    model = StarDist3D(None, name=modelname, basedir=model_basedir)

    predict_opts = {'show_tile_progress': True, 'verbose':True}


    max_size = 224 # for 16GB GPU    
    
    if use_overlap:
        overlap_label = -1
    else:
        overlap_label = None
    
    Y_ = []
    for x in tqdm(X):
        print('Dataset shape: ', x.shape)
        print('Dataset size: ', np.size(x))

        if np.size(x) <= max_size**3: # Limited by 16GB GPU
            y_ = model.predict_instances(x, verbose=True, overlap_label=overlap_label)[0]

        elif np.size(X[0]) <= 250 * 1024 * 1024 or use_overlap: # Limited by 92GB RAM
            n_tiles = tuple(np.max([1, s//max_size]) for s in x.shape)
            print('Num tiles: ', n_tiles)
            y_ = model.predict_instances(x, verbose=True, overlap_label=overlap_label,
                                         n_tiles=n_tiles)[0]
            
        else: # Split into pieces (does not support overlap label, very slow ...)
            min_overlap=(32, 64, 64)
            context=(32, 64, 64)
            
            # block size (246, 704, 704) is too large for 92GB RAM
            # For size 299 block size 196, with overlap & context 32 does not work
            # For size 299 block size 150, with overlap & context 32 should work
            
            # It would be nice to have an automatic estimation what works, such as:
            #block_size = tuple(int(np.ceil(s/2)) + o+2*c for s, o, c in zip(x.shape, min_overlap, context))
            # The stupid way of testing would be just running: 
            # cover_1d = Block.cover(size, block_size, min_overlap, context, grid)
            # ie:
            # cover_1d = Block.cover(299, 150, 32, 32, 1)
            
            # This how I derived for an image with shape (299, 1024, 1024) the block size:
            
            block_size = (150, 704, 704)
            print('Block size: ', block_size)
            y_ = model.predict_instances_big(x,
                                              axes='ZYX',
                                              block_size=block_size,
                                              min_overlap=min_overlap,
                                              context=context, show_progress=True,
                                              verbose=True,
                                              n_tiles=tuple(np.max([1, s//max_size]) for s in x.shape))[0]
        Y_.append(y_)


    if not output_dir.is_dir():
        os.makedirs(output_dir)

    for y, x_path in tqdm(zip(Y_, filelist_x)):
        imsave(output_dir / x_path.name, y, compress=9)
  

      
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar='MODEL', type=str)
    parser.add_argument('dataset_path', metavar='DATASET', type=str)
    parser.add_argument('output_path', metavar='OUTPUT', type=str)
    parser.add_argument('--overlap', action='store_true')
    args = parser.parse_args()
    
    main(args.model_path, args.dataset_path, args.output_path, args.overlap)
