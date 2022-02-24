from email.policy import default
from genericpath import exists
from tifffile import imread
from stardist.matching import  matching_dataset
import numpy as np
from pathlib import Path
import argparse

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
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

def main(output_file, pred_path, gt_path, z_cutoff=None, pattern="*.tif"):

    logger.info(f' Pattern = {pattern}')
    Y_pred_paths = sorted(pred_path.glob(pattern))
    Y_true_paths = sorted(gt_path.glob(pattern))

    for y_pred_path in Y_pred_paths:
        logger.info(y_pred_path)

    for y_true_path in Y_true_paths:
        logger.info(y_true_path)
    
    Y_pred = [imread(str(p)) for p in Y_pred_paths]
    Y_true = [imread(str(p)) for p in Y_true_paths]
    
    if z_cutoff is not None:

        Y_pred = [y_pred[:np.min([z_cutoff, y_pred.shape[0]])] for y_pred in Y_pred]
        Y_true = [y_true[:np.min([z_cutoff, y_true.shape[0]])] for y_true in Y_true]

    results, features = calculateAccuracy(Y_true, Y_pred)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(str(output_file), results, delimiter=',', header=','.join(features))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', metavar='OUTPUT', type=Path)
    parser.add_argument('pred_path', metavar='PREDICTION', type=Path)
    parser.add_argument('gt_path', metavar='GT', type=Path)
    parser.add_argument('--z-cutoff', metavar='ZCUTOFF', type=int, default=None)
    parser.add_argument('--pattern', type=str, default='*.tif')
    args = parser.parse_args()
    
    main(args.output_file, args.pred_path, args.gt_path, args.z_cutoff, args.pattern)

