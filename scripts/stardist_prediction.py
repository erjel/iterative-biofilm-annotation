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
import tensorflow as tf

from time import time

import sys
#sys.path.append(r'C:\Users\Eric\src\stardist_mpcdf')
sys.path.append(r'D:\Eric\stardist_mpcdf')
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

def parse_args():
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group("input")
    data.add_argument('input_folder', metavar='DATASET', type=str)
    data.add_argument('--input-pattern', required=False, default='*.tif')
    data.add_argument('--intp-factor', required=False, type=int, default=None)
    data.add_argument('--overview_plane', action='store_true', default=False)

    model = parser.add_argument_group('model')
    model.add_argument('model_path', metavar='MODEL', type=str)
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--probs', action='store_true')

    output = parser.add_argument_group('output')
    output.add_argument('output_path', metavar='OUTPUT', type=str)
    output.add_argument('--output-name', type=str, default='{file_path}/{model_name}/{file_name}{file_ext}')
  
    return parser, parser.parse_args()

def main():
    parser, args = parse_args()

    model_path = Path(args.model_path)
    input_folder = Path(args.input_folder)
    
    print(f'Input folder: {args.input_folder}')
    print(f'Use pattern: {args.input_pattern}')
    print(f'Output folder: {args.output_path}')
    print(f'Use pattern: {args.output_name}')
    
    X_filenames = sorted(input_folder.glob(args.input_pattern))

    print(f'Found {len(X_filenames)}')

    not args.overview_plane or print('Remove overview planes!')

    args.intp_factor is None or print(f'Interpolate z direction by x{args.intp_factor}')

    axis_norm = (0, 1, 2)

    if args.overlap:
        overlap_label = -1
    else:
        overlap_label = None

    #max_size = 224 # for 16GB GPU
    max_size = 126 # for 11GB GPU
    #max_size = 112 # for 4GB GPU

    
    if tf.__version__.startswith('2'):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    
    print(f'Load model {model_path.name}')
    model = StarDist3D(None, name=model_path.name, basedir=str(model_path.parent))

    for file_in in tqdm(X_filenames):

        file_out = Path(args.output_path) / args.output_name.format (
            file_path = str(file_in.relative_to(args.input_folder).parent),
            file_name = file_in.stem,
            file_ext = file_in.suffix,
            model_name = Path(args.model_path).name
        )

        img = imread(file_in)
        
        if args.overview_plane:
            img = imgs[1:]

        img_shape = img.shape
        
        if args.intp_factor is not None:
            factor = args.intp_factor
            img_shape = (img_shape[0]*factor ,  *img_shape[1:])
            img = ImageInterpolation(img, factor, img_shape)

        print(f'Image shape: {img_shape}')
        img = normalize(img, 1, 99.8, axis=axis_norm)
        
        predict_opts = {'show_tile_progress': True, 'verbose':True}

        _axes         = model._normalize_axes(img, None)
        _axes_net     = model.config.axes
        _permute_axes = model._make_permute_axes(_axes, _axes_net)
        _axes_net     = model.config.axes
        _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')

        if np.size(img) <= max_size**3:
            n_tiles = None
        else:
            n_tiles = tuple(np.max([1, s//max_size]) for s in img.shape)
            print('Num tiles: ', n_tiles)

        prob, dist = model.predict(img, n_tiles=n_tiles)
        y_ = model._instances_from_prediction(_shape_inst, prob, dist, overlap_label=overlap_label, verbose=True)[0]

        if args.probs:
            prop_out = file_out.parent / 'probs' / file_out.name
            prop_out.parent.mkdir(parents=True, exist_ok=True)
            imsave(prop_out, prob, compress=9)
        
        file_out.parent.mkdir(parents=True, exist_ok=True)
        imsave(file_out, y_ ,compress=9)

    return
    
if __name__ == '__main__':
    main()