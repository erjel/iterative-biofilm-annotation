from tifffile import imread
from stardist.matching import  matching_dataset
import numpy as np
from pathlib import Path
import argparse

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

def main(output_file, pred_path, gt_path, z_cutoff=None):
    Y_pred_paths = sorted(pred_path.glob('test/images/*.tif'))
    Y_true_paths = sorted(gt_path.glob('test/masks/*.tif'))
    
    Y_pred = [imread(str(p)) for p in Y_pred_paths]
    Y_true = [imread(str(p)) for p in Y_true_paths]
    
    if z_cutoff is not None:

        Y_pred = [y_pred[:np.min([z_cutoff, y_pred.shape[0]])] for y_pred in Y_pred]
        Y_true = [y_true[:np.min([z_cutoff, y_true.shape[0]])] for y_true in Y_true]

    results, features = calculateAccuracy(Y_pred, Y_true)
    
    np.savetxt(str(output_file), results, delimiter=',', header=','.join(features))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', metavar='OUTPUT', type=str)
    parser.add_argument('pred_path', metavar='PREDICTION', type=Path)
    parser.add_argument('gt_path', metavar='GT', type=Path)
    parser.add_argument('--z-cutoff', metavar='ZCUTOFF', type=int, default=None)
    args = parser.parse_args()
    
    main(args.output_file, args.pred_path, args.gt_path, args.z_cutoff)

