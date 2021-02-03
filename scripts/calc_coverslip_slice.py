from tifffile import imread, TiffFile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import concurrent

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('z_std_csv', type=str)
    parser.add_argument('cover_slip_slice_csv', type=str)
    parser.add_argument('training_data_folders', type=str, nargs='+')

    return parser, parser.parse_args()

def main():
    parser, args = parse_args()

    basepaths = [Path(p) for p in args.training_data_folders]
    std_output_path = Path(args.z_std_csv)
    argmax_output_path = Path(args.cover_slip_slice_csv)

    pattern = 'Pos*/*_ch1_*.tif*'
    min_z=16
    from posixpath import join
    lowChannel='ch2'
    gtChannel='ch1'

    image_paths = [f.parts for d in basepaths for f in ( d ).glob(pattern)]

    z_heights = [int(parts[-1].split('Nz')[-1].split('.tif')[0]) for parts in image_paths]

    source_paths = [join(*parts[:-1], parts[-1].replace(gtChannel, lowChannel)) for parts, Nz in zip(image_paths, z_heights) if Nz >= min_z ]

    if not Path(std_output_path).is_file():

        def compute_mean_std_prctile(path):
            im = imread(str(path))
            # Cut away overview plane and do not care about std values higher than 20 planes
            im = im[1:21]

            im_ = im.reshape(-1, 1024*1024)
            # The threshold is required due to 'hot pixels' which occure in 1:10 of all stacks
            threshs = np.percentile(im_, 99.99, axis=1, keepdims=True)
            im_ = im_.astype('float')
            im_[im_>threshs] = np.nan

            return np.nanstd(im_, axis=1)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_mean_std_prctile, p) for p in source_paths]
            concurrent.futures.wait(futures)

        stds = [f.result()  for f in futures]
        
        """# Simple alternative:
        stds = []

        for p in tqdm(source_paths):
            im = imread(str(p))
            im = im[1:]
            im_ = im.reshape(-1, 1024*1024)
            threshs = np.percentile(im_, 99.99, axis=1, keepdims=True)
            im_ = im_.astype('float')
            im_[im_>threshs] = np.nan
            stds.append(np.nanstd(im_, axis=1))
        """
        
        stds = np.array(stds)
        std_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(std_output_path, stds)
    else:
        stds = np.genfromtxt(std_output_path)

        # Based on the shape of the standard deviation per slice, we expect:
    # - a steep increase for the first slices
    # - a peak
    # - a slow decrease
    # The end of the steep increase can be spot by linearly extrapolating the expected standard deviation.
    # The first value which is below the measured value, the standard deviation did not increase as much as expected.
    # So we are close to the peak.

    stds_argmax = []
    for std in stds:
        intpl = std[0:-2] + 2*np.diff(std[:-1])

        # + 2 is necessary since we can start comparing only after the second std measurement
        # So the stds_argmax is the true index in the stack (without the overview plane)
        stds_argmax.append(np.argwhere(intpl - std[2:] > 0)[0, 0] + 2)

    df = pd.DataFrame(np.asarray([source_paths, stds_argmax]).transpose(), columns=['Path', 'std_argmax'])
    df = df.astype({'Path':'str', 'std_argmax':'uint8'})

    argmax_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(argmax_output_path)

    return

if __name__ == '__main__':
    main()