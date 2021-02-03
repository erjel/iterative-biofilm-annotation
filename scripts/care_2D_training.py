from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

# imports for data preparation
from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches

# imports for training
from csbdeep.utils import axes_dict,  plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data

from csbdeep.models import Config, CARE

# imports for prediction
from csbdeep.utils import Path
from csbdeep.io import save_tiff_imagej_compatible

# custom imports (limiting number of gpus)
import os
import tensorflow as tf

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('output_model', type=str)
    parser.add_argument('trainings_npz', type=str)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=3982)

    return parser, parser.parse_args()

def main():
    parser, args = parse_args()

    np.random.seed(args.seed)

    model_path = Path(args.output_model)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    (X,Y), (X_val,Y_val), axes = load_training_data(args.trainings_npz, validation_split=0.1, verbose=True)

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=1000, train_batch_size=1, unet_n_first=32, train_epochs=500)
    print(config)
    vars(config)

    model = CARE(config, model_path.name, basedir=str(model_path.parent))

    history = model.train(X,Y, validation_data=(X_val,Y_val))

    return

if __name__ == '__main__':
    main()


