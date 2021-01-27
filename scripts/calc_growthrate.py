import numpy as np
import pandas as pd
from tifffile import imread
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('output_csv', type=str)
    parser.add_argument('tracks_csv', type=str)
    parser.add_argument('prediction_folder', type=str)
    parser.add_argument('--prediction_file_pattern', type=str, default='*_frame{frame:06d}*.tif')
    parser.add_argument('--factor', type=float, default=2)

    return parser, parser.parse_args()

def main():
    
    parser, args = parse_args()

    tracks = np.genfromtxt(args.tracks_csv, delimiter=',')

    
    df = pd.DataFrame(tracks, columns=['track_id', 'frame', 'z', 'y', 'x'])

    df['seg_id'] = -1

    root_path = Path(args.prediction_folder)


    for frame in tqdm(df['frame'].unique()):
        print(int(frame)+1)
        candidates = sorted(root_path.glob(args.prediction_file_pattern.format(frame=int(frame+1))))
        print(candidates)
        assert(len(candidates) == 1)
        filename = candidates[0]
        im = imread(str(filename))
        
        belongs_to_frame = df['frame'] == frame
        df_ = df[['z', 'y', 'x']][belongs_to_frame]
        print(len(df_), np.max(im))
        values = (df_.values * args.factor).astype('int')
        z = values[:, 0]
        y = values[:, 1]
        x = values[:, 2]
        
        label_ = im[z, y, x]
        df.at[belongs_to_frame, 'seg_id'] = label_
        
        props = regionprops(im)
        area = np.asarray([p.area for p in props])
        
        _, inv_order = np.unique(label_, return_inverse=True)
        df.at[belongs_to_frame, 'volume'] = area[inv_order]

    output_name = Path(output_csv)

    if not output_name.parent.is_dir():
        output_name.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_name)


if __name__ == '__main__':
    main()