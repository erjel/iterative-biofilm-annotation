# Post-processing according to 
# Ji Zhang et al.:
# “BCM3D 2.0: Accurate Segmentation of Single Bacterial Cells in Dense Biofilms Using Computationally Generated Intermediate Image Representations”
# (bioRxiv, September 6, 2022),
# https://doi.org/10.1101/2021.11.26.470109.

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from tifffile import imread, imwrite
import numpy as np

from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, relabel_sequential
from scipy.ndimage import binary_erosion, grey_dilation
from scipy.stats import iqr



def normalize(img_stack: np.ndarray, low=3, high=99.9) -> np.ndarray:
    p_low, p_high = np.percentile(img_stack, [low, high])
    return (img_stack - p_low) / (p_high - p_low)

def seeded_watershed(ws_map: np.ndarray, thresh: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    seeds_labeled = label(ws_map > thresh)
    
    props = regionprops(seeds_labeled)
    volumes = np.array([p.area for p in props]) 
    labels = np.array([p.label for p in props])

    lut = np.arange(seeds_labeled.max() + 1, dtype=seeds_labeled.dtype)
    lut[labels[volumes < 30]] = 0

    seeds_labeled = lut[seeds_labeled]

    result = watershed(
        -ws_map,
        markers = seeds_labeled,
        mask = mask,
    )

    return result

def delete_small_objects(label_stack: np.ndarray, thresh: float) -> np.ndarray:
    props = regionprops(label_stack)
    volumes = np.array([p.area for p in props])
    labels = np.array([p.label for p in props])

    lut = np.arange(label_stack.max()+1, dtype=label_stack.dtype)
    lut[labels[volumes < thresh]] = 0
    
    return lut[label_stack]

def post_processing(output_tif: Path, edt_tif: Path, bdy_tif: Path) -> None:
    target1 = imread(edt_tif)
    target2 = imread(bdy_tif)

    ## Post processing of "distance to nearest cell exterior"

    # SI: "Predicted ‘distance to nearest cell exterior’ images were first normalized
    # by a simple percentile-based normalization method"
    target1_normalized = normalize(target1)

    # SI: "After applying Otsu-thresholding to the ‘distance to nearest cell exterior’
    #  image to obtain a binary image (Figure S3b), connected voxel clusters can be
    #  isolated and identified assingle cell objects by labeling connected regions"
    thresh = threshold_otsu(target1_normalized)
    target1_binarized_ = target1_normalized > thresh

    # SI: "To split clusters that are only connected by one or two voxels, the
    #  boundary voxels of each object were set to zero before labeling connected"
    target1_binarized = binary_erosion(target1_binarized_)
    target1_labels = label(target1_binarized)

    # SI: "After labeling, the erased boundary voxels were added back to each object"
    target1_labels = grey_dilation(target1_labels, size=(2,2,2))

    # SI: "A conservative size-exclusion filter was applied: small objects with volume
    #  smaller than the radius cubed of the targeted cells were considered background
    #  noise and filtered out."

    # NOTE(erjel): 
    # V(vx) = 400nm/px x 63nm/px x 63nm/px = 0.00158 um^3/vx
    # expected cell radius vibrio cholerae (Hartmann et al. Nature Physics (2019)):
    # r = 0.2775 um 
    # -> r^3 = 0.0213 um^3
    # -> V(threshold) = r^3/V(vx) ~ 14.2 vx 

    r_thresh = 14.2
    target1_labels = delete_small_objects(target1_labels, r_thresh)


    ## Post-processing with "proximity enhanced cell boundary"

    # SI: "Objects that need further processing were found by evaluating its volume
    #  and solidity, i.e., the volume to convex volume ratio. Here, volume is
    #  defined as the number of voxels occupied by an object. Convex volume is
    #  defined as the number of voxels of a convex hull, which is the smallest
    #  convex polygon that encloses an object. The upper limit was found by using
    #  the interquartile rule, i.e. the upper limit is quartile 3 (Q3) plus 1.5
    #  times interquartile range (IQR). If an object's volume or solidity is
    #  larger than the upper limit, it will be singled out for further processing.

    props = regionprops(target1_labels)
    solidity = np.array([p.solidity for p in props])
    volume = np.array([p.area for p in props])
    labels = np.array([p.label for p in props])

    v_thresh = np.percentile(volume, 75) + 1.5 * iqr(volume)
    s_thresh = np.percentile(solidity, 25) - 1.5 * iqr(solidity)

    # NOTE(erjel): 
    # * The radius threshold is quite low, maybe make it a little higher?
    # * Why are there so many 1.0 value sin the solidtiy?
    # * It does not make sense to me to use the solidty threshold on 3 quartile of
    #   volume/ convex volume. It is probably better to apply it on the lower rang
    undersegmented_labels = labels[(solidity < s_thresh) & (volume > v_thresh)]

    lut = np.zeros(np.max(target1_labels) + 1,dtype=target1_labels.dtype)
    lut[undersegmented_labels] = undersegmented_labels
    target1_undersegmented = lut[target1_labels]

    # SI: "All these objects together generate a new binary image"
    mask = target1_undersegmented > 0

    # SI: "CNN-produced ‘proximity enhanced cell boundary’ images were first
    #  normalized by the same percentile-based normalization method"
    target2_normalized = normalize(target2)

    # SI: "Specifically, we generated a difference map by subtracting the
    #  ‘proximity enhanced cell boundary’ image from the ‘distance to nearest cell
    #  exterior’ image and then set all negative valued voxels to zero"
    factor = target1_normalized - target2_normalized
    factor[factor < 0] = 0

    # SI: "This difference map was then multiplied by the binary image generated in
    #  Step 1"
    labels_filtered = mask * factor


    # SI: "Then, the resulting image was segmented by seeded watershed. Seeds were
    #  obtained by Otsu thresholding of the difference map and seeds with a volume
    #  smaller than 30 voxels were removed"
    thresh = threshold_otsu(labels_filtered)

    ws_result = seeded_watershed(labels_filtered, thresh, mask)

    # SI: "These new objects were again evaluated by their volume and solidity;"
    ws_props = regionprops(ws_result)

    ws_solidity = np.array([p.solidity for p in ws_props])
    ws_volume = np.array([p.area for p in ws_props])
    ws_labels = np.array([p.label for p in ws_props])

    undersegmented_labels1 = ws_labels[(ws_solidity < s_thresh) & (ws_volume > v_thresh)]

    # SI: "if there still exist unmatched objects, a multi-level Otsu thresholding
    #  will be applied to further  generate seeds"

    if len(undersegmented_labels1) == 0:
        print('No undersegmented labels left, directly merge label images!')
        raise NotImplementedError

    # SI: "Seeds were extracted by using the third and the fourth threshold successively"
    _, _, thresh1, thresh2 = threshold_multiotsu(labels_filtered, 5)


    # SI: "The same size filter was used to remove unreasonable small seeds"
    lut = np.zeros(ws_result.max()+1, dtype=ws_result.dtype)
    lut[undersegmented_labels1] = undersegmented_labels1
    mask1 = lut[ws_result] > 0

    ws_result1 = seeded_watershed(labels_filtered, thresh1, mask1)

    ws_props2 = regionprops(ws_result1)

    ws_solidity2 = np.array([p.solidity for p in ws_props2])
    ws_volume2 = np.array([p.area for p in ws_props2])
    ws_labels2 = np.array([p.label for p in ws_props2])

    undersegmented_labels2 = ws_labels2[(ws_solidity2 < s_thresh) & (ws_volume2 > v_thresh)]

    if len(undersegmented_labels2) == 0:
        print('No undersegmented labels left, directly merge label images!')
        raise NotImplementedError

    lut = np.zeros(ws_result1.max()+1, dtype=ws_result1.dtype)
    lut[undersegmented_labels2] = undersegmented_labels2
    mask2 = lut[ws_result1] > 0

    ws_result2 = seeded_watershed(labels_filtered, thresh2, mask2)


    ## Combine the watershed segmentations 

    results = [target1_labels, ws_result, ws_result1, ws_result2]

    for result in results:
        assert not any(np.unique(target1_labels[result > 0]) == 0)

    result_labels = results[0]

    # SI: "A conservative size-exclusion filter was applied: small objects with
    #  volume 10 times smaller than the upper limit volume were considered
    #  unreasonable small parts and filtered out."

    for result in results[1:]:
        result = delete_small_objects(result, v_thresh/10)
        result_labels += (result_labels.max() * (result > 0)) + result

    result_labels, _, _ = relabel_sequential(result_labels)

    # SI: "Since the ‘distance to nearest cell exterior’ images were confined
    #  to the cell interior, we dilated each object by 1-2 voxels to increase the
    #  cell volumes using standard morphological dilation"

    result_labels = grey_dilation(result_labels, size=(3,3,3))

    imwrite(output_tif, result_labels)

    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('edt_dir', type=Path)
    parser.add_argument('cell_bdy_dir', type=Path)
    parser.add_argument('--input-pattern', type=str, default='*.tif')

def main() -> None:
    args = parse_args()

    edt_files = sorted(args.input_dir.glob(args.input_pattern))
    cell_bdy_files = sorted(args.input_dir.glob(args.input_pattern))

    for edt_tif, bdy_tif in zip(edt_files, cell_bdy_files):
        assert edt_tif.name == bdy_tif.name
        post_processing(
            args.output_dir / edt_tif.name,
            edt_tif,
            bdy_tif,
        )

if __name__ == '__main__':
    main()






