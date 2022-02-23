from distutils import filelist
from pathlib import Path
from tifffile import imsave, imread
from cellpose import models
import os
import mxnet as mx
from tqdm import tqdm
import argparse
import logging

import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(model_path, dataset_path, output_path):

    logger.info(f'Look for data in {str(dataset_path)}')
    
    assert dataset_path.is_dir(), f'{str(dataset_path)} is not a directory!'

    filelist_x = sorted(dataset_path.glob('*.tif'))
    X = [imread(p) for p in filelist_x]

    logger.info(f'Found {len(X)} images in the data directory')

    output_dir = output_path

    if not output_dir.is_dir():
        os.makedirs(output_dir)

    logger.info(f'Search for model in "{model_path}"')

    model_path = sorted((model_path / 'models').glob('cellpose_*'))
    logger.info(f'Found {len(model_path)} models')

    assert len(model_path) == 1, f"model_path contains {len(model_path)} objects"
    model_path = model_path[0]
    logger.info(f'Use model "{model_path}"')

    szmean = 15.
    device = mx.gpu()
    batch_size = 8
    channels = [0, 0]
    do_3D = True
    
    if do_3D:
        X = [x[..., None] for x in X]

    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(model_path),
        batch_size=batch_size,
        diam_mean=szmean,
        net_avg=True,
        device=device,
        unet=False)

    logger.info(f'input data has shape {X[0].shape}')

    Y_pred = model.eval(X, do_3D=do_3D, flow_threshold=None, channels=channels)[0]

    for y, x_path in tqdm(zip(Y_pred, filelist_x)):
        imsave(output_dir / x_path.name, y, compress=9)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar='MODEL', type=Path)
    parser.add_argument('dataset_path', metavar='DATASET', type=Path)
    parser.add_argument('output_path', metavar='OUTPUT', type=Path)
    args = parser.parse_args()
    
    main(
        args.model_path,
        args.dataset_path,
        args.output_path
    )