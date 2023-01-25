import numpy as np
from itertools import product
import os
from tifffile import imsave, imread
from glob import glob
from socket import gethostname
import re
from platform import system
from pathlib import Path
from scipy.ndimage import affine_transform

def getRootDir():
    #TODO(erjel): You should have moved past hard-coded directories -.-
    dataset_root = 'training_data'
    return dataset_root

def sliceToShape(data_tuple, output_shape=(100, 100, 100), verbose=False):
    num_dim = tuple(s // out_s for s, out_s in zip(data_tuple[0][0].shape, output_shape))
    print('Number of slices: ', num_dim)
    
    offsets = tuple((s - i * out_s) // i for s, i, out_s in zip(data_tuple[0][0].shape, num_dim, output_shape))
    
    o1, o2, o3 = offsets
    #print('offsets = ', o1, o2, o3)
    i, j, k = output_shape

    
    verbose and [print('({}, {}, {}): '.format(l, m, n),
               '{:3d}:{:3d}, '.format((l*o1) + l*i, (l*o1) + (l+1)*i),
               '{:3d}:{:3d}, '.format((m*o2) + m*j, (m*o2) + (m+1)*j),
               '{:3d}:{:3d}'.format((n*o3) + n*k, (n*o3) + (n+1)*k))
             for l, m, n in product(*[range(i) for i in num_dim])]

    
    data_tuple = tuple(
            [x[(l*o1) + l*i:(l*o1) + (l+1)*i,
               (m*o2) + m*j:(m*o2) + (m+1)*j,
               (n*o3) + n*k:(n*o3) + (n+1)*k]
                 for l, m, n in product(*[range(i) for i in num_dim])
             for x in X]
        for X in data_tuple)
    
    return data_tuple

def readDataset(datasetName, verbose=False):

    dataset_root = getRootDir()

    data_classes = ['images', 'masks']
    data_purposes = ['train', 'valid', 'test']
    
    verbose and print(os.path.join(dataset_root, datasetName, data_purposes[0], data_classes[0], '*.tif'))
    
    X, Y = tuple(
        {data_purpose:
            [imread(x) for x in sorted(glob(os.path.join(dataset_root, datasetName, data_purpose, data_class, '*.tif')))]
        for data_purpose in data_purposes} for data_class in data_classes)

    return X, Y

def writeDatasetFolders(X, dataset_dir, datafunction, datatype, filenames=None, verbose=True):

    dataset_root = getRootDir()
    
    im_dir = os.path.join(dataset_dir, datafunction, datatype)
    
    verbose and print('Write dataset to "{}"'.format(im_dir))

    if not os.path.isdir(im_dir):
        os.makedirs(im_dir)

    for i, im in enumerate(X):
        if filenames is None:
            im_name = 'im{}.tif'.format(i)
        else:
            im_name = filenames[i]
        im_path = os.path.join(im_dir, im_name)
        imsave(im_path, im)

    return

def correctFullStackOffset(label_vol, raw_vol, verbose=False):
    conds = [np.roll(label_vol, offset, 0) > 0 for offset in range(-1, 2)]
    max_offset = np.argmax([np.sum(raw_vol[c]) for c in conds])
    verbose and print(max_offset)
    if max_offset == 0:
        verbose and print('Offset between files exist!')
        label_vol = label_vol[1:]
        raw_vol = raw_vol[:-1]

    if np.argmax(np.sum(raw_vol, axis=(1, 2))) == 0:
        verbose and print('overview plane detected!')
        raw_vol[0] = 0
        
    if verbose:
        conds = [np.roll(label_vol, offset, 0) > 0 for offset in range(-1, 2)]      
        max_idx = np.argmax([np.sum(raw_vol[c]) for c in conds])
        print(np.argmax([np.sum(raw_vol[c]) for c in conds]) == 1)
    
    return label_vol, raw_vol


def getFullStacks(datatype='raw'): # in the long run, I need to create proper datasets (i.e. just plain text files?)
    
    dataset_root = Path(getRootDir())

    image_path = dataset_root / 'BiofilmQ_{}'.format(datatype)

    annotation_path =Path(__file__).parent / 'data'/ 'napari_annotation'
    original_label_paths = sorted(annotation_path.glob('**/*0?/*0L.tif'))
    
    image_files_paths = sorted(image_path.glob('images/*.tif'))
    label_files_paths = sorted(image_path.glob('masks/*.tif'))
    
    print('found {} labeled files'.format(len(original_label_paths)))
    
    assert(len(original_label_paths) == len(image_files_paths) and 
           len(original_label_paths) == len(label_files_paths))
    
    for i in range(len(original_label_paths)):
        print('{} -> {}'.format([original_label_paths[i].parts[j] for j in [-4, -1]], image_files_paths[i].parts[-1]))
      
    return image_files_paths, label_files_paths, original_label_paths


def checkFullStackData(images, labels):
    for l, im in zip(labels, images):
        assert(len(l.shape) == len(im.shape)) # same number of dimensions
        for i in range(len(l.shape)):         # same number of elements in each dimension
            assert(l.shape[i] == im.shape[i])

        conds = [np.roll(l, offset, 0) > 0 for offset in range(-1, 2)]
        assert(np.argmax([np.sum(im[c]) for c in conds]) == 1)  # no z-offset

        assert(np.argmax(np.sum(im, axis=(1, 2))) != 0) # no overview plane
        
    return images, labels

def cutFullStackIntoSlices(images, labels, filelist=None, start_2D_dim=0, end_2D_dim=None):
    
    assert(len(labels) == len(images))
    if filelist is None:
        assert(len(filelist) == len(images))
    
    if end_2D_dim is None:
        end_2D_dim = len(images[0].shape)
    
    dim_range = range(start_2D_dim, end_2D_dim)
    slice_string = ['xy', 'xz', 'yz']
    slice_string = [slice_string[i] for i in dim_range]
    slice_string = '_'.join(slice_string)
    
    X = []
    Y = []
    filenames = []
    for k, (x, y) in enumerate(zip(images, labels)):
        for j in dim_range:
            for i in range(x.shape[j]):
                indices = (slice(None), )*j + (i, )
                X.append(x[indices])
                Y.append(y[indices])
                if filelist is None:
                    label = 'im{}'.format(k)
                else:
                    label = filelist[k]
                    
                filenames.append('{}_{}_{}.tif'.format(label, slice_string, i))
    
    return X, Y, filenames

def ImageInterpolation(image, d, new_shape):

    trans_matrix = np.diag([1/d, 1, 1])
    rescaled = affine_transform(image, trans_matrix, output_shape=new_shape, order=3)

    return rescaled

if __name__ == '__main__':
	pass