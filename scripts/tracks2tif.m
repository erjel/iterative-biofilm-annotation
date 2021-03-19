function tracks2tif(output_path, tracks_csv_path, prediction_folder)
    biofilmq_path = fullfile(getenv('Home'), 'src', 'BiofilmQ', 'includes');
    addpath(genpath(biofilmq_path));
	
	biofilmq_path_CPU_SERVER = 'D:\Eric\bacbq';
	addpath(genpath(biofilmq_path_CPU_SERVER));
	
	biofilmq_path_GPU_SERVER = 'C:\Users\Eric\src\biofilmq\includes';
	addpath(genpath(biofilmq_path_GPU_SERVER));
    
    if ~isfolder(output_path)
        mkdir(output_path)
    end

    
    % Read track csv file
    tracks = readmatrix(tracks_csv_path);
    unique_track_ids = unique(tracks(:, 1));
    new_track_ids = unique_track_ids(randperm(length(unique_track_ids)));
    tracks(:, 1) = new_track_ids(tracks(:, 1)+1);
    
    segmentation_files = dir(fullfile(prediction_folder, '*.tif'));
    
    scaling_factors = [4, 1, 1];
        
    for i = 1:numel(segmentation_files)
        segmentation_file_path = fullfile( ...
            segmentation_files(i).folder, segmentation_files(i).name);
        
        labelimage = imread3D(segmentation_file_path);
        
        new_object_ids = [0; nan(max(labelimage(:)), 1)];
        
        tracks_in_frame = tracks(tracks(:, 2) == i-1, :);
        centroids = round(tracks_in_frame(:, 3:end) .* scaling_factors);
        Z = centroids(:, 1);
        Y = centroids(:, 2);
        X = centroids(:, 3);
        Track_IDs = tracks_in_frame(:, 1);
        
        for j = 1:size(tracks_in_frame, 1)
            x = X(j);
            y = Y(j);
            z = Z(j);
            id = labelimage(y, x, z);
            if ~(id == 0)
                new_object_ids(id+1) = Track_IDs(j);
            end
        end
        
        not_assigned = isnan(new_object_ids);
        max_assigned_id = max( new_object_ids(~not_assigned));
        not_assigned_ids = max_assigned_id+1:max_assigned_id+sum(not_assigned);
        
        new_object_ids(not_assigned) = not_assigned_ids;
         
        output_file = fullfile(output_path, sprintf('frame%06d.tif', i));
        fprintf('%s %d\n', output_file, isfolder(output_path));
        
        imwrite3D(new_object_ids(labelimage+1), output_file);
        
    end
end