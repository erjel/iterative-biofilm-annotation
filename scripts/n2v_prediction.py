
from n2v.models import N2V
from tifffile import imread, imsave
from pathlib import Path

import os

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('output_tiff', type=str)
    parser.add_argument('input_tiff', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--has_overview', action='store_true')

    return parser, parser.parse_args()

def main():

    parser, args = parse_args()

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id) 
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

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