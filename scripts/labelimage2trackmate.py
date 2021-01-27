from tifffile import imread
from pathlib import Path
from skimage.measure import regionprops
from jinja2 import Template
import numpy as np

from tqdm import tqdm

from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('--int_data_path', type=str, default=r"T:\to_delete\care_model_eva-v1-dz400-care_rep1_v4.zip")
    parser.add_argument('--input_folder', type=str, default=r'T:\to_delete\debug')
    parser.add_argument('--output_xml', type=str, default=r"Y:\Eric\prediction_test\data\interim\tracking\debug_full.xml")

    return parser, parser.parse_args()


def main():
    
    int_data_shape = (54, 1024, 1024)
    int_data_scaling = np.array([0.400, 0.063, 0.063])
    int_seg_scaling = np.array([0.100, 0.063, 0.063])
    scaling_factors = int_data_scaling / int_seg_scaling

    print('start stardist2trackmate.py')

    parser, args = parse_args()


    int_data_path = Path(args.int_data_path)
    input_folder = Path(args.input_folder)
    output_xml = Path(args.output_xml)
    
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    seg_files = sorted(input_folder.glob('*.tif'))

    total_num = 0
    data = {
        "total_num": -1,
        "frames": [],
        'int_data_file': int_data_path.name,
        "int_data_folder": str(int_data_path.parent),
        'width': int_data_shape[2],
        'height': int_data_shape[1],
        'n_sices': int_data_shape[0],
        'n_frames': len(seg_files),
        'dx': int_data_scaling[2],
        'dy': int_data_scaling[1],
        'dz': int_data_scaling[0],
        'width_': int_data_shape[2]-1,
        'height_': int_data_shape[1]-1,
        'n_sices_': int_data_shape[0]-1,
        'n_frames_': len(seg_files)-1,
    }

    for frame_id, seg_file in enumerate(tqdm(seg_files)):

        img = imread(str(seg_file))

        props = regionprops(img)
        frame = {"spots": [], 'frame_id':frame_id}
        for p in props:
            spot = {}
            total_num += 1
            spot['id'] = total_num
            spot['quality'] = 1
            spot['time'] = 1.0*frame_id
            spot['max_intensity'] = 1
            spot['frame'] = frame_id
            spot['median_intensity'] = 1
            spot['visibility'] = 1
            spot['mean_intensity'] = 1
            spot['total_intensity'] = 1
            spot['estimated_diameter'] = 20
            spot['radius'] = 7
            spot['snr'] = 1
            spot['x'] = p.centroid[2] / scaling_factors[2]
            spot['y'] = p.centroid[1] / scaling_factors[1]
            spot['std'] = 1
            spot['contrast'] = 1
            spot['manual_color'] = 1
            spot['min_intensity'] = 1
            spot['z'] = p.centroid[0] / scaling_factors[0]
            frame['spots'].append(spot)

        data['frames'].append(frame)            

    data['total_num'] = total_num

    with open('resources/template_Trackmate.xml', 'r') as f:
        template = ''.join(f.readlines())

    tmp = Template(template)

    with open(output_xml, 'w') as f:
        f.write(tmp.render(data))

    return

if __name__ == '__main__':
    main()