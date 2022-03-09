
from argparse import ArgumentParser, Namespace
from tifffile import imread, imsave
from stardist.matching import  matching
from typing import Optional
import numpy as np
from pathlib import Path
from skimage.segmentation import relabel_sequential


def color_cells(volume, idcs):
    """
    background = 0
    not in idcs = 1
    in idcs = 2
    """
    idcs = np.asarray(idcs)
    lut = np.ones(np.max(volume)+1)
    lut[idcs] = 2
    lut[0] = 0
    return lut[volume]


def create_fn_fp_tifs(
    output_fn: Path,
    output_fp: Path,
    prediction_path: Path,
    ground_truth_path: Path,
    thresh: Optional[float] = 0.5,
    remove_pred_slice: Optional[bool] = False
) -> None:
    
    Y_pred_paths, Y_true_paths = prediction_path, ground_truth_path
    
    if remove_pred_slice:
        Y_pred = imread(str(Y_pred_paths))
        Y_true = imread(str(Y_true_paths))[1:]
    else:
        Y_pred, Y_true = tuple(imread(str(paths)) for paths in [Y_pred_paths, Y_true_paths]) 
    Y_pred, Y_true = tuple(relabel_sequential(Y_)[0] for Y_ in [Y_pred, Y_true])

    m = matching(Y_true, Y_pred, thresh=0.5, report_matches=True)

    matching_dict = m._asdict()
    matched_pairs = np.asarray(matching_dict['matched_pairs'])
    matched_tps = np.asarray(matching_dict['matched_tps'])
    
    Y_true_ids = matched_pairs[matched_tps, 0]
    Y_pred_ids = matched_pairs[matched_tps, 1]

    Y_true_, Y_pred_ = color_cells(Y_true, Y_true_ids), color_cells(Y_pred, Y_pred_ids)
    
    imsave(output_fn, Y_true_, compress=9)
    imsave(output_fp, Y_pred_, compress=9)

    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('output_fn_tif', type=Path)
    parser.add_argument('output_fp_tif', type=Path)
    parser.add_argument('prediction_tif', type=Path)
    parser.add_argument('ground_truth_tif', type=Path)
    parser.add_argument('remove_pred_slice', type=str)

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    create_fn_fp_tifs(
        args.output_fn_tif,
        args.output_fp_tif,
        args.prediction_tif,
        args.ground_truth_tif,
        args.remove_pred_slice == 'True',
    )

    return

if __name__ == '__main__':
    main()