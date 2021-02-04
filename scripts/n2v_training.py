from n2v.models import N2VConfig, N2V
import numpy as np
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import os
from pathlib import Path

from argparse import ArgumentParser
from utils import use_gpu

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('model_path', type=str)
    parser.add_argument('input_folder', type=str)
    parser.add_argument('--seed', type=str, default='15082')

    return parser, parser.parse_args()

@use_gpu
def main():
    parser, args = parse_args()

    model_path = Path(args.model_path)

    datagen = N2V_DataGenerator()

    imgs = datagen.load_imgs_from_directory(directory = args.input_folder, filter='*_ch1_*.tif', dims='ZYX')

    imgs = [i[:, 1:np.max([34, round(n/len(imgs)*55)]), 300:700, 300:700, :] for n, i in enumerate(imgs)]

    patch_shape = (32, 64, 64)
    patches = datagen.generate_patches_from_list(imgs_, shape=patch_shape)
    del imgs

    idcs = np.arange(len(patches))

    np.random.shuffle(idcs)

    idcs_train = idcs[:round(len(idcs)*0.9)]
    idcs_valid = idcs[round(len(idcs)*0.9):]
    del idcs

    X = patches[idcs_train]
    X_val = patches[idcs_valid]
    del patches

    config = N2VConfig(X, unet_kern_size=3, 
                   train_steps_per_epoch=int(X.shape[0]/128),train_epochs=200, train_loss='mse', batch_norm=True, 
                   train_batch_size=4, n2v_perc_pix=0.198, n2v_patch_shape=(32, 64, 64), 
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)

    vars(config)

    # We are now creating our network model.
    model = N2V(config=config, name=model_path.name, basedir=str(model_path.parent))

    history = model.train(X, X_val)

    return

if __name__ == '__main__':
    main()