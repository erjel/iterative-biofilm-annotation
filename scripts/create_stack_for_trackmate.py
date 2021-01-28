from tifffile import imread, imwrite
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
	shape = img.shape

	stack = np.zeros((len(img_files),shape[0], 1, *shape[1:], 1), dtype=np.uint8)

	for i, img_file in enumerate(tqdm(img_files),):
		if i != 0: # first image already loaded
			img = imread(str(img_file))

		img = img - np.min(img)
		img = img / np.max(img)
		img[img < 0 ] = 0
		img[img > 1] = 1
		img = img * 2**8 - 1
		img = img.astype(np.uint8)

		stack[i] = img[:, None, :, :, None]

	imwrite(output_stack, stack, imagej=True, metadata={'axes': 'TZCYXS'}, compression=('DEFLATE', 1))
	
	
	return
	
if __name__ == '__main__':
	main()