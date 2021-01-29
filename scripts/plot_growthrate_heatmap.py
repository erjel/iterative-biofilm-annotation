import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('output_fig', type=str)
    parser.add_argument('input_csv', type=str)
    

    return parser, parser.parse_args()

def main():
    parser, args = parse_args()

    df = pd.read_csv(Path(args.input_csv))

    df[['mean_z', 'mean_y', 'mean_x']] = \
        df.groupby(['frame'])[['z', 'y', 'x']].transform('mean')

    df['distance'] = np.sum(
            (
                df[['z', 'y', 'x']].values 
                    - df[['mean_z', 'mean_y', 'mean_x']].values
            )**2,
         axis=1)

    nbins = 50

    v = df.groupby([
        pd.cut(df["frame"], np.linspace(df['frame'].min(), df['frame'].max(), nbins)),
        pd.cut(df['distance'], np.linspace(df['distance'].min(), df['distance'].max(), nbins))
    ])['volume_diff'].mean()

    heatmap = v.unstack()

    ax = sns.heatmap(heatmap)
    f = ax.get_figure()

    output_name = Path(args.output_fig)
    output_name.parent.mkdir(parents=True, exist_ok=True)

    f.savefig(output_name)
    return

if __name__ == '__main__':
    main()