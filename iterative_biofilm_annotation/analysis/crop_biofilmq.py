from argparse import ArgumentParser
from pathlib import Path
from tifffile import imread, imwrite

CROP = (slice(10, 139), slice(255, 512), slice(255,512))

def main(output: Path, input: Path) -> None:
    im = imread(input)
    output.mkdir(parents=True, exist_ok=True)
    imwrite(Path(output) / 'im0.tif', im[CROP])

    return None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()

    main(args.output, args.input)