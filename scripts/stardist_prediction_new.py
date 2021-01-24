import tensorflow as tf

#if tensorflow2:

import tensorflow as tf

if tf.__version__.startswith('2'):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

from time import time

import sys
sys.path.append(r'C:\Users\Eric\src\stardist_mpcdf')
#sys.path.append(r'D:\Eric\stardist_mpcdf')
#sys.path.append(r'D:\Users\Eric\src\stardist_mpcdf')

from stardist_mpcdf.data import ImageInterpolation

def allocateOnEmptyGPU():
    import os
    import re
    import numpy as np
    from subprocess import check_output

    nvidia_smi_output = check_output(['nvidia-smi']).decode("utf-8")
    memory_matches = re.findall('\d+MiB\s*/\s*\d+MiB', nvidia_smi_output)
    memory_string = [match.split('MiB')[0] for match in memory_matches]
    gpu_memory_usage = list(map(int, memory_string))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin(gpu_memory_usage))
    print('Run on GPU with ID: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    return

#allocateOnEmptyGPU()

def main(model_path, dataset_path, output_path, use_overlap, has_overview_plane=False):

    stardist_mpcdf_home = Path(os.getcwd()).parent

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    dataset_name = dataset_path.stem
    modelname = model_path.name
    model_basedir = model_path.parent

    print(f'Read {dataset_path}: ', end='')
    tic = time()
    assert(dataset_path.is_file())
    X = [imread(str(dataset_path))]
    
    print('{:.2f}s'.format((time() - tic)))
    
    not has_overview_plane or print('Remove overview plane ...')
    
    X = [x[1:] if has_overview_plane else x for x in X]
    
    output_dir = output_path
    if use_overlap:
        filelist_x = [Path(str(dataset_path).replace('.tif', 'O.tif'))]
    else:
        filelist_x = [Path(str(dataset_path).replace('.tif', 'P.tif'))]

    dz = 400
    factor = int(dz/100)    
    
    print(f'Interpolate z direction by x{factor}: ', end='')
    tic = time()
    
    if not dz is None:
        for i, x in enumerate(X):
            
            new_shape = (x.shape[0]*factor ,  *x.shape[1:])
            X[i] = ImageInterpolation(x, factor, new_shape)
            
    print('{:.2f}s'.format((time() - tic)))
    
    print(f'New dataset shape: {new_shape}')
        

    print(f'Normalize dataset:', end="")
    tic = time()
    axis_norm = (0, 1, 2)
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    
    print('{:.2f}s'.format((time() - tic)))

    print(dataset_name)

    print('Dataset length: ', len(X))

    print('Load model "{}"'.format(modelname))
    model = StarDist3D(None, name=modelname, basedir=model_basedir)

    predict_opts = {'show_tile_progress': True, 'verbose':True}

    #max_size = 224 # for 16GB GPU
    max_size = 126 # for 11GB GPU
    max_size = 112 # for 4GB GPU
    max_size = 96
    
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

        else:
            n_tiles = tuple(np.max([1, s//max_size]) for s in x.shape)
            print('Num tiles: ', n_tiles)
            y_ = model.predict_instances(x, verbose=True, overlap_label=overlap_label,
                                         n_tiles=n_tiles)[0]
        """  
        else: # Split into 4 pieces (does not support overlap label, very slow ...)
            min_overlap=(32, 64, 64)
            context=(32, 64, 64)
            block_size = tuple(int(np.ceil(s/2)) + o+2*c for s, o, c in zip(x.shape, min_overlap, context))
            print('Block size: ', block_size)
            y_ = model.predict_instances_big(x,
                                              axes='ZYX',
                                              block_size=block_size,
                                              min_overlap=min_overlap,
                                              context=context, show_progress=True,
                                              verbose=True,
                                              n_tiles=tuple(np.max([1, s//max_size]) for s in x.shape))[0]
        """
        Y_.append(y_)


    if not output_dir.is_dir():
        os.makedirs(output_dir)

    for y, x_path in tqdm(zip(Y_, filelist_x)):
        imsave(output_dir / x_path.name, y, compress=9)
  

      
    return

filelist = sorted(Path(r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5').glob('*_ch1_frame0000??_*.tif'))

len(filelist)

for f in filelist:
    main(r"models\eva-v1_dz400_rep1", str(f), 'predictions', False, True)

