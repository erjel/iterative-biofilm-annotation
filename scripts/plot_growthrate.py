import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from argparse import ArgumentParser


def parse_args():

    parser = ArgumentParser()
    parser.add_argument('output_fig', type=str)
    parser.add_argument('input_df_csv', type=str)
    return parser, parser.parse_args()


def main():

    parser, args = parse_args()

    df = pd.read_csv(Path(args.input_df_csv))

    f, ax = plt.subplots()


    for t_id in df['track_id'].unique():
        df_ = df[['frame', 'volume']][(df['track_id'] == t_id) & (df['seg_id'] != 0)]

        ax.plot(df_['frame'], df_['volume'])
        
    #ax.set_ylim(0, 5000)
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Volume')

    output_name = Path(args.output_fig)

    if not output_name.parent.is_dir():
        os.makedirs(output_name.parent)

    f.savefig(output_name)
    return

if __name__ == '__main__':
    main()