
from n2v.models import N2V
from tifffile import imread, imsave
from pathlib import Path

import os

from argparse import ArgumentParser
from utils import use_gpu


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('output_tiff', type=str)
    parser.add_argument('input_tiff', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--has_overview', action='store_true')

    return parser, parser.parse_args()

@use_gpu
def main():

    parser, args = parse_args()

    model_path = Path(args.model_path)

    im = imread(args.input_tiff)
    if args.has_overview:
        im = im[1:]

    model = N2V(config=None, name=model_path.name, basedir=str(model_path.parent))
    pred = model.predict(im, axes='ZYX', n_tiles=(2,4,4))
    imsave(args.output_tiff, pred)

    return

if __name__ == '__main__':
    main()