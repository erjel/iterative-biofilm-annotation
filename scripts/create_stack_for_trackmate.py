from tifffile import imread
from pathlib import Path
from tqdm import tqdm
import numpy as np

from argparse import ArgumentParser

def parse_args():

	parser = ArgumentParser()
	
	parser.add_argument('output_stack', type=str)
	parser.add_argument('input_folder', type=str)
	

	return parser, parser.parse_args()


def main():
	import numpy as np
	parser, args = parse_args()
	
	input_folder = Path(args.input_folder)
	output_stack = Path(args.output_stack)
	output_stack.parent.mkdir(parents=True, exist_ok=True)

	img_files = sorted(input_folder.glob('*.tif'))

	img = imread(str(img_files[0]))

	shape = np.asarray(img.shape, dtype='int')
	factor = 2
	shape = (shape/ factor) * np.asarray([len(img_files), 1, 1])

	Tracking_stack = np.zeros(shape.astype('int'), dtype=np.int8)
	print(Tracking_stack.shape)

	from tifffile import imsave
	import os
	from skimage.measure import regionprops
	import numpy as np

	for i, img_file in enumerate(tqdm(img_files)):

		img = imread(str(img_file))
		shape = img.shape

		props = regionprops(img)
		
		centroids = np.asarray([p.centroid for p in props])
		centroids = centroids / factor
		
		centroids = np.rint(centroids)
		centroids = centroids.astype(np.int16)
			
		for z, y, x in centroids:
			Tracking_stack[int(z+shape[0]/2*i), y, x] = 2**8-1

	imsave(output_stack , Tracking_stack)

	print(shape[0] // 2, len(img_files))
	
	return
	
if __name__ == '__main__':
	main()