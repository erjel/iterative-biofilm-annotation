from argparse import ArgumentParser, Namespace
from pathlib import Path

from scipy.ndimage import zoom
from tifffile import imread, imwrite


def downsample_tif(output_tif: Path, input_tif: Path, output_dz: float, input_dz: float) -> None:

    im = imread(input_tif)
    scaling = (input_dz / output_dz, 1, 1)
    out = zoom(im, scaling, order=0)

    # Note(erjel): Necessary since stardist Nz differs from biofilmq Nz
    out = out[:-1, :, :]

    imwrite(output_tif, out)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('output_tif', type=Path)
    parser.add_argument('input_tif', type=Path)
    parser.add_argument('output_dz', type=float)
    parser.add_argument('input_dz', type=float)

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    downsample_tif(
        args.output_tif,
        args.input_tif,
        args.output_dz,
        args.input_dz
    )

    return

if __name__ == "__main__":
    main()