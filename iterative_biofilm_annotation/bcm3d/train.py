from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Tuple

from csbdeep.data import shuffle_inplace
from csbdeep.models import Config, CARE
from csbdeep.utils import normalize
import numpy as np
from tifffile import imread
from tqdm import tqdm

from iterative_biofilm_annotation.unet.utils import crop

def train(
    basedir: Path,
    modelname: str,
    dataset: str,
    target: str,
    patch_size: Tuple[int],
    epochs: int,
    steps: int,
) -> None:

    # load and crop out central patch (for simplicity)
    X_train = [crop(imread(x), patch_size) for x in sorted(glob(f'training_data/{dataset}/train/images/*.tif'))]
    Y_train = [crop(imread(y), patch_size) for y in sorted(glob(f'training_data/{dataset}/train/{target}/*.tif'))]

    X_valid = [crop(imread(x), patch_size) for x in sorted(glob(f'training_data/{dataset}/valid/images/*.tif'))]
    Y_valid = [crop(imread(y), patch_size) for y in sorted(glob(f'training_data/{dataset}/valid/{target}/*.tif'))]

    # normalize input image
    X_train = [normalize(x,1,99.8) for x in tqdm(X_train)]
    X_valid = [normalize(x,1,99.8) for x in tqdm(X_valid)]

    # convert to numpy arrays
    X_train, Y_train = np.expand_dims(np.stack(X_train),-1), np.expand_dims(np.stack(Y_train), -1)
    X_valid, Y_valid = np.expand_dims(np.stack(X_valid),-1), np.expand_dims(np.stack(Y_valid), -1)

    # shuffle data
    shuffle_inplace(X_train, Y_train, seed=0)
    shuffle_inplace(X_valid, Y_valid, seed=0)

    axes = 'SZYXC'
    config = Config(axes, n_channel_in=1, n_channel_out=1, train_steps_per_epoch=steps)
    model = CARE(config, modelname, basedir=basedir)
    history = model.train(X_train,Y_train, validation_data=(X_valid,Y_valid), epochs=epochs)

    return


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('modeldir', type=Path)
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('train_patch_size', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100)

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    train(
        basedir = args.modeldir.parent,
        modelname = args.modeldir.name,
        dataset = args.dataset_name,
        target = args.target,
        patch_size = tuple(int(s) for s in args.train_patch_size.split('x')),
        epochs = args.epochs,
        steps = args.steps,
    )

if __name__ == '__main__':
    main()