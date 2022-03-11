function tif2vtk(output_vtk, value_tif, label_tif, x, y, z)

    addpath(genpath(fullfile('external', 'BiofilmQ', 'includes')));
    
    if nargin < 4
        x =  512;
    end
    if nargin < 5
        y =  512;
    end
    if nargin < 6
        z = 25;
    end

    [~, vtk_label, ~] = fileparts(value_tif);
        
    volume = imread3D(value_tif);
    
    pred_volume = imread3D(label_tif);
    
    mask_x = false(size(pred_volume));
    mask_y = false(size(pred_volume));
    mask_z = false(size(pred_volume));

    mask_x(x:end, :, :) = true;
    mask_y(:, y:end, :) = true;
    mask_z(:, :, z:end) = true;

    mask = ~(mask_x & mask_y & mask_z);

    mask = uint16(mask);


    pred_volume_masked = pred_volume .* mask;
    
    unique_ids = unique(pred_volume_masked);
    
    relabel_indices = 0:unique_ids(end);
    relabel_indices(~ismember(relabel_indices, unique_ids)) = 0;
    
    % +1 since backgound has value 0 and has to be mapped to index 1
    pred_volume_masked = relabel_indices(pred_volume_masked+1); 
    
    objects = conncomp(pred_volume_masked);


    vals = zeros(1, objects.NumObjects);
    for i = 1:objects.NumObjects
        if ~isempty(objects.PixelIdxList{i})
            val = volume(objects.PixelIdxList{i}(1));

            if all(volume(objects.PixelIdxList{i}) == val)
                vals(i) = val;
            else
                disp(unique(volume(objects.PixelIdxList{i})))
                disp('Prediction and False-Postive estimate do not fit together!')
                    return
            end
        end
    end
    
    [~, filename] = fileparts(output_vtk);
    fprintf('%s\n', filename)
    fprintf('%s:      %d\n', vtk_label, sum(vals(:) == 1))
    fprintf('not %s:  %d\n', vtk_label, sum(vals(:) == 2))
    fprintf('bg:      %d\n', sum(vals(:) == 0))

    objects.stats = regionprops(objects);

    vals = num2cell(vals);

    [objects.stats.(vtk_label)] = vals{:};

    objects2VTK(output_vtk, objects, 0.5);
    
end