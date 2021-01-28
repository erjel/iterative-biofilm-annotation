
addpath(genpath('D:\Users\Eric\src\BiofilmQ\includes'));

im_seg = imread3D("Y:\Eric\prediction_test\data\interim\predictions\care\eva-v1-dz400-care_rep1\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame000080_Nz54.tif");

im_care = imread3D("Y:\Eric\prediction_test\data\interim\care\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame000080_Nz54.tif");

im_raw = imread3D("Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame000080_Nz54.tif");

im_huy = imread3D("Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\kdv1502R_5L_30ms_300gain002_pos5_ch2_frame000080_Nz54.tif");

im_biofilmq = imread3D('T:\test.tif');

im_raw(:, :, 1) = [];
im_huy(:, :, 1) = [];

im_seg = im_seg(182:182+570, 276:276+496, :);
im_raw = im_raw(182:182+570, 276:276+496, :);
im_huy = im_huy(182:182+570, 276:276+496, :);
im_care = im_care(182:182+570, 276:276+496, :);


params.scaleUp = false;

im_care = zInterpolation(im_care, 0.063, 0.4, params);
im_raw = zInterpolation(im_raw, 0.063, 0.4, params);
im_seg = zInterpolation_nearest(im_seg, 0.063, 0.1, params);
im_huy = zInterpolation(im_huy, 0.063, 0.4, params);
im_biofilmq = zInterpolation(im_biofilmq, 0.063, 0.4, params);

zSlicer(im_care, 'parula', 'clim', [400, 1000])
zSlicer(im_raw, 'parula')
zSlicer(im_huy, 'parula')
zSlicer(im_biofilmq, 'parula')

% zSlicer(im_seg)