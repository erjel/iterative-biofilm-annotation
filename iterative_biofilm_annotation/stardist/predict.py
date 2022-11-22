#!/usr/bin/env python
import argparse
from bz2 import compress
from genericpath import exists
import logging
import os
from pathlib import Path
from time import time
import sys

from csbdeep.utils import normalize
import numpy as np
from stardist.models.model3d import StarDist3D
import tensorflow as tf
from tifffile import imsave, imread
from tqdm import tqdm

from data import ImageInterpolation

try:
    from merge_stardist_masks.naive_fusion import naive_fusion
    from stardist.rays3d import rays_from_json
    MERGE_AVAILABLE = True
except ImportError:
    MERGE_AVAILABLE = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ['OMP_PLACES'] = "threads"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
    output.add_argument('--overwrite', action='store_true', default=False)
    output.add_argument('--use-merge', action='store_true', default=False)
  
    return parser, parser.parse_args()

def main():
    parser, args = parse_args()

    use_merge = args.use_merge

    if use_merge and not MERGE_AVAILABLE:
        logger.info(
            'To use the merging post-processing install "merge-stardist-masks". \n' + \
            'The prediction will proceed with the merging post-processing'
        )
        use_merge = MERGE_AVAILABLE

    model_path = Path(args.model_path)
    input_folder = Path(args.input_folder)
    
    logger.info(f'Input folder: {args.input_folder}')
    logger.info(f'Use pattern: {args.input_pattern}')
    logger.info(f'Output folder: {args.output_path}')
    logger.info(f'Use pattern: {args.output_name}')
    
    X_filenames = sorted(input_folder.glob(args.input_pattern))

    logger.info(f'Found {len(X_filenames)}')

    not args.overview_plane or logger.info('Remove overview planes!')

    args.intp_factor is None or logger.info(f'Interpolate z direction by x{args.intp_factor}')

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
        if len(physical_devices) > 0:
            config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    
    logger.info(f'Load model {model_path.name}')
    model = StarDist3D(None, name=model_path.name, basedir=str(model_path.parent))

    for file_in in tqdm(X_filenames):

        file_out = Path(args.output_path) / args.output_name.format (
            file_path = str(file_in.relative_to(args.input_folder).parent),
            file_name = file_in.stem,
            file_ext = file_in.suffix,
            model_name = Path(args.model_path).name if not use_merge else f'{Path(args.model_path).name}_merge'
        )
        
        prop_out = file_out.parent / 'probs' / file_out.name
        
        if not args.overwrite:
            if args.probs and file_out.is_file() and prop_out.is_file():
                continue
            elif not args.probs and file_out.is_file():
                continue
            
        logger.debug(f'Read file "{str(file_in)}"')        
        img = imread(file_in)
        
        if args.overview_plane:
            img = img[1:]

        img_shape = img.shape
        
        if args.intp_factor is not None:
            factor = args.intp_factor
            img_shape = (img_shape[0]*factor ,  *img_shape[1:])
            img = ImageInterpolation(img, factor, img_shape)

        logger.info(f'Image shape: {img_shape}')
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
            logger.info(f'Num tiles: {n_tiles}')

        
        if use_merge:
            prob, dist = model.predict(img, n_tiles=n_tiles)
            rays = rays_from_json(model.config.rays_json)
            # TODO(erjel): Save the dist and props on disk and use a separate job for naive fusison?
            save_intermediate_results = False
            if save_intermediate_results:
                file_out.parent.mkdir(parents=True, exist_ok= True)
                logger.info('Save probabilites')
                imsave(file_out.parent / (file_out.stem + '_prob.tif'), prob)
                logger.info('Save distances')
                imsave(file_out.parent / (file_out.stem + '_dist.tif'), dist)
                logger.info('Perform naive fusion')
            y_ = naive_fusion(dist-1, prob, rays, grid=model.config.grid, prob_thresh=model.thresholds.prob)

        else:
            logger.info('Start sparse prediction for standard stardist')
            prob, dist, points = model.predict_sparse(img, n_tiles=n_tiles)
            logger.info(f'Probability shape: {prob.shape}')
            y_, details = model._instances_from_prediction(_shape_inst, prob, dist, points=points, overlap_label=overlap_label, verbose=True)
        
            details_out = file_out.parent / 'details' /file_out.stem
            details_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(details_out, details)

        logger.info(f'Naive output shape: {y_.shape}')
        y_ = y_[tuple(slice(0, s) for s in img_shape)]
        logger.info(f'Cropped output shape: {y_.shape}')


        if args.probs:
            prop_out = file_out.parent / 'probs' / file_out.name
            prop_out.parent.mkdir(parents=True, exist_ok=True)
            imsave(prop_out, prob, compress=9)
        
        file_out.parent.mkdir(parents=True, exist_ok=True)
        imsave(file_out, y_ ,compress=9)

    return
    
if __name__ == '__main__':
    main()
