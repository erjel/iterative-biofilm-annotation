import concurrent
from pathlib import Path
from typing import Tuple


from edt import edt
from tifffile import imread, imwrite
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, grey_closing

def bcm3d_targets(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    euclidian_dist = edt(labels)
    binary_mask = labels > 0
    
    next_cell_dist_ = np.zeros_like(euclidian_dist)
    
    label_vals = np.unique(labels)
    label_vals = label_vals[label_vals != 0]
    
    for v in label_vals:
        mask = labels == v
        
        selection = euclidian_dist[mask]
        euclidian_dist[mask] = selection / selection.max()
        
        labels_ = np.ones_like(next_cell_dist_)
        labels_[binary_mask] = 0
        labels_[mask] = 1
        proximity = edt(labels_)
        next_cell_dist_[mask] = 1/proximity[mask]        
    
    cell_ext_dist = euclidian_dist ** 3
    next_cell_dist = binary_mask - euclidian_dist
    
    # Note(erjel): Paper describes Gaussian blur with simga = (5,5,5)
    cell_ext_dist = gaussian_filter(cell_ext_dist, sigma=(2,2,2))
        
    next_cell_dist *= next_cell_dist_
    
    # Note(erjel): Unclear kernel size for grey closing in paper
    next_cell_dist = grey_closing(next_cell_dist, size=(2,2,2))
    # Note(erjel): Paper describes Gaussian blur with simga = (5,5,5)
    next_cell_dist = gaussian_filter(next_cell_dist, sigma=(2,2,2))
    
    return cell_ext_dist, next_cell_dist

def bcm3d_wrapper(label_path: Path) -> None:
    labels = imread(label_path)
    
    cell_ext_dist, next_cell_dist = bcm3d_targets(labels)
    
    output_dir_ext = label_path.parent.parent / 'target_bacm3d_1'
    output_dir_ext.mkdir(parents=True, exist_ok=True)
    
    output_dir_next = label_path.parent.parent / 'target_bacm3d_2'
    output_dir_next.mkdir(parents=True, exist_ok=True)
    
    imwrite(output_dir_ext / label_path.name, cell_ext_dist)   
    imwrite(output_dir_next / label_path.name, next_cell_dist)  
    
    return

def run(f, my_iter):
    l = len(my_iter)
    with tqdm(total=l) as pbar:

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(f, arg): arg for arg in my_iter}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                arg = futures[future]
                results[arg] = future.result()
                pbar.update(1)

            return results
    
if __name__ == '__main__':
    label_paths = sorted(Path('training_data/patches-semimanual-raw-64x128x128').glob('**/masks/*.tif'))
    
    print(f'Process {len(label_paths)} file')
    
    run(bcm3d_wrapper, label_paths)
    print('Done!')