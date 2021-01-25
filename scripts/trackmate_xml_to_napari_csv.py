from xml.dom import minidom
import numpy as np

from argparse import ArgumentParser

def parse_args():

	parser = ArgumentParser()
	
	parser.add_argument('input_path_xml', type=str)
	parser.add_argument('output_path_csv', type=str)

	return parser, parser.parse_args()

def main():

	parser, args = parse_args()

	mydoc = minidom.parse(args.input_path_xml)

	data = []

	for i, f in enumerate(mydoc.getElementsByTagName('Tracks')[0].getElementsByTagName('particle')):
		for d in f.getElementsByTagName('detection'):
			d.attributes.keys()
			data.append([i, *[float(d.attributes[k].value) for k in ['t', 'z', 'y', 'x']]])


	
	data = np.asarray(data)

	np.savetxt(args.output_path_csv, data, delimiter=',')

	return
    
if __name__ == '__main__':
    main()