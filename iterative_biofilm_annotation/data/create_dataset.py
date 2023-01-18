import argparse
import logging
from pathlib import Path
import re

from stardist import fill_label_holes

from tifffile import imread, imsave
from skimage.segmentation import relabel_sequential
from tqdm import tqdm
import numpy as np

from iterative_biofilm_annotation.stardist.data import sliceToShape, writeDatasetFolders, ImageInterpolation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main(
    output_dir: Path,
    label_stack_dir: Path, 
    image_stack_dir: Path, 
    pattern: str, 
    patch_size: str,
    ) -> None:
    """Calculate patches for tif stacks
    """

    label_paths = sorted(label_stack_dir.glob('*'))
    logger.info(f'Found {len(label_paths)} files in {label_stack_dir}')
    [logger.debug(label_path) for label_path in label_paths]

    label_paths = [l for l in label_paths if re.match(pattern, str(l))]
    logger.info(f'Found {len(label_paths)} files match "{pattern}":')
    [logger.info(label_path) for label_path in label_paths]
    
    X = []
    Y = []

    for label_path in label_paths:
        image_stack_path = image_stack_dir / label_path.name.replace('labels', 'raw')

        assert image_stack_path.is_file(), \
            f'{image_stack_path} does not exist!'

        label = imread(label_path)
        raw_file = imread(image_stack_path)

        assert label.shape == raw_file.shape, \
            f'{label.shape=}\n{raw_file.shape=}'
        
        X.append(raw_file)
        Y.append(label)


    Y = [relabel_sequential(y)[0] for y in tqdm(Y)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert re.match('\d+x\d+x\d+', patch_size), \
        f'"{patch_size}" does not match pattern'
    X, Y = sliceToShape([X, Y], output_shape=tuple(int(s) for s in patch_size.split('x')))

    patch_size = tuple(int(s) for s in patch_size.split('x'))

    X = [x for x in X if x.shape == patch_size]
    Y = [x for x in Y if x.shape == patch_size]

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

    logger.info('number of images: %3d' % len(X))
    logger.info('- training:       %3d' % len(X_trn))
    logger.info('- validation:     %3d' % len(X_val))
    logger.info('- test:           %3d' % len(X_test))

    datasets = [X_val, X_trn, X_test, Y_val, Y_trn, Y_test]
    dataPurposes = ['valid', 'train', 'test']*2
    dataTypes = ['images']*3 + ['masks']*3

    for dataset, dataPurpose, dataType in zip(datasets, dataPurposes, dataTypes):
        writeDatasetFolders(dataset, output_dir, dataPurpose, dataType)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('label_stack_dir', type=Path)
    parser.add_argument('image_stack_dir', type=Path)
    parser.add_argument('pattern', type=str)
    parser.add_argument('patchsize', type=str)
    args = parser.parse_args()


    main(
        args.output_dir,
        args.label_stack_dir,
        args.image_stack_dir,
        args.pattern,
        args.patchsize,
    )
