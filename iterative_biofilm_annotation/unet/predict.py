from argparse import ArgumentParser, Namespace
import math
from pathlib import Path
from typing import List, Optional, Tuple

from csbdeep.utils import normalize
import numpy as np
from skimage.measure import label
from stardist.big import BlockND
from tifffile import imread, imwrite
from tqdm import tqdm

from utils import SegModel

def predict(
    modelname: str,
    basedir: Path,
    input_data_list: List[Path],
    output_dir: Path,
    context: Optional[Tuple[int]] = (16, 16, 16),
    chunk_size: Optional[Tuple[int]] = (48, 96, 96),
    gpu_batch_size: Optional[int] = 1,
 ) -> None:

    model = SegModel(None, modelname, basedir=basedir)
    axes = 'ZYX'

    def process(x):
        return model.predict(x, axes=axes)

    print(f'Number of files to predict: {len(input_data_list)}')

    for input_path in input_data_list:

        img = imread(input_path)

        chunk_size_ = tuple(min([s, c]) for s,c in zip(chunk_size, img.shape)) 
        chunk_size_ = tuple(
            int(math.floor(s / c)*c) if c > 0 else 1 for s, c in zip(chunk_size, context)
        )

        blocks = BlockND.cover(
            shape=img.shape,
            axes=axes,
            block_size=chunk_size_,
            min_overlap=(0,) * img.ndim,
            context=context,
        )

        img_processed_tiled = np.empty(img.shape + (3,), dtype=np.float32)

        # START https://python-forum.io/thread-12022.html
        import itertools 
        def chunker_longest(iterable, chunksize):
            return itertools.zip_longest(*[iter(iterable)] * chunksize)

        gpu_cycles = math.ceil(len(blocks) / gpu_batch_size)

        for block_list in tqdm(chunker_longest(blocks, gpu_batch_size), total=gpu_cycles):
            input = [b.read(img, axes=axes) for b in block_list if b is not None]
            input = [normalize(i, 1, 99.8) for i in input]
            # TODO(erjel): Train UNets with SZYXC axes to benefit from batch inference
            #input = np.array(input)
            #result = process(input)

            input = input[0]
            result = [process(input)]

            [b.write(img_processed_tiled, b.crop_context(r, axes=axes)) for r, b in zip(result, block_list) if b is not None]

        lbl_pred = label(img_processed_tiled[...,1] > 0.7)

        output_path = output_dir / modelname / input_path.name

        output_path.parent.mkdir(parents=True, exist_ok=True)

        imwrite(output_path, lbl_pred, compression ='zlib')

    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('output_folder', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('input_folder', type=Path)
    parser.add_argument('--input-pattern', type=str, default='*.tif')

    return parser.parse_args()

def main() -> None:
    args = parse_args() 

    print(args.model_dir)   

    predict(
        modelname = args.model_dir.name,
        basedir = args.model_dir.parent,
        input_data_list = sorted(args.input_folder.glob(args.input_pattern)),
        output_dir = args.output_folder,
    )

    return

if __name__ == '__main__':
    main()