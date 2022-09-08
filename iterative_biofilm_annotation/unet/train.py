# Modified from 
# https://github.com/CSBDeep/CSBDeep/blob/master/examples/other/technical.ipynb

"""
BSD 3-Clause License

Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Tuple

from csbdeep.utils import normalize
from csbdeep.data import shuffle_inplace
import numpy as np
from tifffile import imread
from tqdm import tqdm


from utils import crop, to_3class_label, SegModel, SegConfig



def train(
    basedir: Path,
    modelname: str,
    dataset: str,
    patch_size: Tuple[int],
    epochs: int,
    steps: int,
    ) -> None:

    # load and crop out central patch (for simplicity)
    X_train       = [crop(imread(x), patch_size) for x in sorted(glob(f'training_data/{dataset}/train/images/*.tif'))]
    Y_train_label = [crop(imread(y), patch_size) for y in sorted(glob(f'training_data/{dataset}/train/masks/*.tif'))]

    # load and crop out central patch (for simplicity)
    X_valid       = [crop(imread(x), patch_size) for x in sorted(glob(f'training_data/{dataset}/valid/images/*.tif'))]
    Y_valid_label = [crop(imread(y), patch_size) for y in sorted(glob(f'training_data/{dataset}/valid/masks/*.tif'))]

    # normalize input image and convert label image to 3-class segmentation mask
    X_train = [normalize(x,1,99.8) for x in tqdm(X_train)]
    Y_train = [to_3class_label(y) for y in tqdm(Y_train_label)]

    # normalize input image and convert label image to 3-class segmentation mask
    X_valid = [normalize(x,1,99.8) for x in tqdm(X_valid)]
    Y_valid = [to_3class_label(y) for y in tqdm(Y_valid_label)]

    # convert to numpy arrays
    X_train, Y_train, Y_train_label = np.expand_dims(np.stack(X_train),-1), np.stack(Y_train), np.stack(Y_train_label)

    # convert to numpy arrays
    X_valid, Y_valid, Y_valid_label = np.expand_dims(np.stack(X_valid),-1), np.stack(Y_valid), np.stack(Y_valid_label)

    config = SegConfig(n_channel_in=1, n_channel_out=3, unet_depth=2)
    model = SegModel(config, modelname, basedir=str(basedir))
    model

    # shuffle data
    shuffle_inplace(X_train, Y_train, Y_train_label, seed=0)
    shuffle_inplace(X_valid, Y_valid, Y_valid_label, seed=0)

    # for demonstration purposes: training only for a very short time here
    history = model.train(X_train,Y_train, validation_data=(X_valid,Y_valid),
                        lr=3e-5, batch_size=1, epochs=epochs, steps_per_epoch=steps)

    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('modeldir', type=Path)
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('train_patch_size', type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=100)

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    train(
        basedir = args.modeldir.parent,
        modelname = args.modeldir.name,
        dataset = args.dataset_name,
        patch_size = tuple(int(s) for s in args.train_patch_size.split('x')),
        epochs = args.epochs,
        steps = args.steps,
    )

if __name__ == '__main__':
    main()
