from pathlib import Path
from tifffile import imread, imsave
from cellpose import models
import os
import mxnet as mx
from tqdm import tqdm
import argparse

import sys
sys.path.append(str(Path(os.getcwd())))

def main(model_path, dataset_path, output_path):
    
    dataset_name = dataset_path.stem
    modelname = model_path.name
    
    assert(dataset_path.is_file())

    X = [imread(str(dataset_path))]

    output_dir = Path('interference') / dataset_name / modelname

    if not output_dir.is_dir():
        os.makedirs(output_dir)

    model_path = sorted((model_path / 'models').glob('cellpose_*'))
    assert(len(model_path) == 1)
    model_path = model_path[0]

    szmean = 15.
    #device = mx.gpu()
    batch_size = 8
    channels = [0, 0]
    do_3D = True
    
    if do_3D:
        X = [x[..., None] for x in X]

    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(model_path),
        batch_size=batch_size,
        diam_mean=szmean,
        net_avg=True,
        #device=device,
        unet=False)

    print(X[0].shape)

    Y_pred = model.eval(X, do_3D=do_3D, flow_threshold=None, channels=channels)[0]

    filelist_x = [Path(str(dataset_path).replace('.tif', 'P.tif'))]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for y, x_path in tqdm(zip(Y_pred, filelist_x)):
        imsave(output_dir / x_path.name, y, compress=9)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar='MODEL', type=str)
    parser.add_argument('dataset_path', metavar='DATASET', type=str)
    parser.add_argument('output_path', metavar='OUTPUT', type=str)
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    
    main(model_path, dataset_path, output_path)