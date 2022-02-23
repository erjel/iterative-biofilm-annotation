from tifffile import imread
from pathlib import Path
from skimage.measure import regionprops
from jinja2 import Template
import numpy as np
from scipy.ndimage import affine_transform

from tqdm import tqdm

from argparse import ArgumentParser

def labelimage2trackmate(int_data_path, input_folder, output_xml):
    # TODO(erjel): Get rid of hard-coded properties!
    int_data_shape = (54, 1024, 1024)
    int_data_scaling = np.array([0.400, 0.063, 0.063])
    int_seg_scaling = np.array([0.100, 0.063, 0.063])
    scaling_factors = int_data_scaling / int_seg_scaling


    int_data_path = Path(int_data_path)
    im_data = imread(int_data_path)
    input_folder = Path(input_folder)
    output_xml = Path(output_xml)
    
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
    #for frame_id, seg_file in enumerate(tqdm(seg_files[:10])):

        img = imread(str(seg_file))
        
        im_data_ = im_data[frame_id]
        
        new_shape = np.asarray(im_data_.shape)
        new_shape = new_shape * np.array([4, 1, 1])        

        trans_matrix = np.diag([1/scaling_factors[0], 1, 1])
        #TODO(erjel): Niklas mentioned problems with interpolation (use zoom?)
        rescaled = affine_transform(im_data_, trans_matrix, output_shape=new_shape, order=1)
        

        props = regionprops(img, rescaled)
        
        frame = {"spots": [], 'frame_id':frame_id}
        for p in props:
            spot = {}
            total_num += 1
            spot['id'] = total_num
            spot['quality'] = 1
            spot['time'] = 1.0*frame_id
            spot['max_intensity'] = p.max_intensity
            spot['frame'] = frame_id
            spot['median_intensity'] = p.label
            spot['visibility'] = 1
            spot['mean_intensity'] = p.mean_intensity
            spot['total_intensity'] = 1
            spot['estimated_diameter'] = np.cbrt(p.area*3/(4* np.pi))*2
            spot['radius'] = 7
            spot['snr'] = 1
            spot['x'] = p.centroid[2] / scaling_factors[2] # make it to physical units -> requires proper image!
            spot['y'] = p.centroid[1] / scaling_factors[1]
            spot['std'] = 1
            spot['contrast'] = 1
            spot['manual_color'] = p.label
            spot['min_intensity'] = 1
            spot['z'] = p.centroid[0] / scaling_factors[0]
            frame['spots'].append(spot)

        data['frames'].append(frame)            

    data['total_num'] = total_num

    with open('resources/template_TrackMate.xml', 'r') as f:
        template = ''.join(f.readlines())

    tmp = Template(template)

    with open(output_xml, 'w') as f:
        f.write(tmp.render(data))

    return

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('--int_data_path', type=str)
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_xml', type=str)

    return parser, parser.parse_args()


def main():
    print('start stardist2trackmate.py')

    parser, args = parse_args()
    
    labelimage2trackmate(
        args.int_data_path,
        args.input_folder,
        args.output_xml
    )

if __name__ == '__main__':
    main()