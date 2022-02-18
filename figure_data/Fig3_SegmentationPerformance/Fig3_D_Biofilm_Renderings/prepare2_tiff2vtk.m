filename_list = dir('*.tif');


pred_path = {
    '../../chapter_single_cell_segmentation/full_semimanual-huy/test/masks/im0.tif', ...
    '../../chapter_single_cell_segmentation/full_semimanual-huy/test/images_Pos1/2020-08-17_data_seeded_watershed/huy_seeded_watershed.tif', ...
    '../../chapter_single_cell_segmentation/full_semimanual-huy/test/masks/im0.tif', ...
    '../predictions/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep5/datasets/full_semimanual-raw/test/images/im0.tif'};

% manually selected slices:
y =  512;
x =  512;
z = 25;

apply_clip = true;


for i = 1:numel(filename_list)
% for i = 4
    filename =  filename_list(i).name;
    filename_vtk = strrep(filename, 'tif', 'vtk')
    output_path = fullfile(filename_list(i).folder, filename_vtk);
    volume_path = fullfile(filename_list(i).folder, filename);
    
    % quick and dirty ...
    if i == 3
        im_pred = imread3D(pred_path{i});
        im_pred(:, :, 1) = [];
        tmp_filename = strrep(filename, '.tif', '_tmp.tif');
        imwrite3D(im_pred, tmp_filename);
        if apply_clip
            fp2vtk(strrep(filename, '.tif', ''), output_path, volume_path, tmp_filename, x, y, z);
        else
            fp2vtk(strrep(filename, '.tif', ''), output_path, volume_path, tmp_filename);
        end
        system(sprintf('rm -r %s', tmp_filename));
    else
        if apply_clip
            fp2vtk(strrep(filename, '.tif', ''), output_path, volume_path, pred_path{i}, x, y, z);
        else
            fp2vtk(strrep(filename, '.tif', ''), output_path, volume_path, pred_path{i});
        end
    end
end

function fp2vtk(label_string, output_path, volume_path, pred_path, x, y, z)

    apply_clip = ~(nargin == 4);

    addpath(genpath('~/src/bacbq/includes'));

    assert(isfile(pred_path));
    assert(isfile(volume_path));
        
    volume = imread3D(volume_path);
    pred_volume = imread3D(pred_path);
    
    if apply_clip
        mask_x = false(size(pred_volume));
        mask_y = false(size(pred_volume));
        mask_z = false(size(pred_volume));
        
        mask_x(x:end, :, :) = true;
        mask_y(:, y:end, :) = true;
        mask_z(:, :, z:end) = true;
        
        mask = ~(mask_x & mask_y & mask_z);
        
        mask = uint16(mask);
        
        
        pred_volume_masked = pred_volume .* mask;
    else
        pred_volume_masked = pred_volume;
    end
    
    unique_ids = unique(pred_volume_masked);
    relabel_indices = 0:unique_ids(end);
    relabel_indices(~ismember(relabel_indices, unique_ids)) = 0;
    
    
    
    pred_volume_masked = relabel_indices(pred_volume_masked+1); % +1 since backgound has value 0 and has to be mapped to index 1
    
    objects = conncomp(pred_volume_masked);


    vals = zeros(1, objects.NumObjects);
    for i = 1:objects.NumObjects
        if ~isempty(objects.PixelIdxList{i})
            val = volume(objects.PixelIdxList{i}(1));

            if all(volume(objects.PixelIdxList{i}) ==val)
                vals(i) = val;
            else
                disp('Prediction and False-Postive estimate do not fit together!')
                return
            end
        end
    end
    
    [~, filename] = fileparts(output_path);
    fprintf('%s\n', filename)
    fprintf('%s:      %d\n', label_string, sum(vals(:) == 1))
    fprintf('not %s:  %d\n', label_string, sum(vals(:) == 2))
    fprintf('bg:      %d\n', sum(vals(:) == 0))

    objects.stats = regionprops(objects);

    vals = num2cell(vals);

    [objects.stats.(label_string)] = vals{:};

    objects2VTK(output_path, objects, 0.5);
    
end