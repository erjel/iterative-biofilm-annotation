from email.policy import default
from genericpath import exists
from tifffile import imread
from stardist.matching import  matching_dataset
import numpy as np
from pathlib import Path
import argparse

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def calculateAccuracy(Y_true, Y_pred):

    tau_vals = np.arange(0.05, 1, 0.05)
    m = matching_dataset(Y_true, Y_pred, thresh=tau_vals, show_progress=True, parallel=True)

    results = np.zeros((np.size(tau_vals), len(m[0]._asdict().keys())))

    for i in range(results.shape[0]):
        for j, feature in enumerate(m[i]._asdict().keys()):
            val = m[i]._asdict()[feature]
            if not isinstance(val, str):
                results[i, j] = val
                
    features = [s for s in m[0]._asdict().keys()]
    return results, features

def main(output_file, pred_path, gt_path, z_cutoff_gt=None, z_cutoff_pred=None, pattern_gt=None, pattern_pred=None):
    logger.info(f' Prediction directory = {str(pred_path)}')
    logger.info(f' Prediction pattern = {pattern_pred}')
    logger.info(f' Ground truth directory = {str(gt_path)}')
    logger.info(f' Ground truth pattern = {pattern_gt}')

    Y_pred_paths = sorted(pred_path.glob(pattern_pred))
    Y_true_paths = sorted(gt_path.glob(pattern_gt))

    logger.info(f'Prediction images (n={len(Y_pred_paths)}):')
    for y_pred_path in Y_pred_paths:
        logger.info(y_pred_path)

    logger.info(f'GT images (n={len(Y_true_paths)})')
    for y_true_path in Y_true_paths:
        logger.info(y_true_path)
    
    Y_pred = [imread(str(p)) for p in Y_pred_paths]
    Y_true = [imread(str(p)) for p in Y_true_paths]
    
    if z_cutoff_gt is not None or z_cutoff_pred is not None:
        slice_pred = slice(None,) if not z_cutoff_pred else slice(*[int(s) for s in z_cutoff_pred.split(':')])
        slice_gt = slice(None,) if not z_cutoff_gt else slice(*[int(s) for s in z_cutoff_gt.split(':')])

        Y_pred = [y_pred[slice_pred] for y_pred in Y_pred]
        Y_true = [y_true[slice_gt] for y_true in Y_true]

    results, features = calculateAccuracy(Y_true, Y_pred)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(str(output_file), results, delimiter=',', header=','.join(features))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', metavar='OUTPUT', type=Path)
    parser.add_argument('pred_path', metavar='PREDICTION', type=Path)
    parser.add_argument('gt_path', metavar='GT', type=Path)
    parser.add_argument('--z-cutoff-gt', type=str, default=None)
    parser.add_argument('--z-cutoff-pred', type=str, default=None)
    parser.add_argument('--pattern', type=str, default='**/*.tif')
    parser.add_argument('--pattern_gt', type=str, default=None)
    parser.add_argument('--pattern_pred', type=str, default=None)
    args = parser.parse_args()

    pattern_gt = args.pattern_gt if args.pattern_gt is not None else args.pattern
    pattern_pred = args.pattern_pred if args.pattern_pred is not None else args.pattern
    
    main(
        args.output_file,
        args.pred_path,
        args.gt_path,
        z_cutoff_gt=args.z_cutoff_gt,
        z_cutoff_pred=args.z_cutoff_pred,
        pattern_gt=pattern_gt,
        pattern_pred=pattern_pred
    )

