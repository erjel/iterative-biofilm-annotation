function trackmate2vtk(output_path, trackmate_xml, prediction_folder)

    biofilmq_path = fullfile(getenv('Home'), 'src', 'BiofilmQ', 'includes');
    addpath(genpath(biofilmq_path));
	
	biofilmq_path_CPU_SERVER = 'D:\Eric\bacbq';
	addpath(genpath(biofilmq_path_CPU_SERVER));
	
	biofilmq_path_GPU_SERVER = 'C:\Users\Eric\src\biofilmq\includes';
	addpath(genpath(biofilmq_path_GPU_SERVER));
    
    if ~isfolder(output_path)
        mkdir(output_path)
    end
    

    % Read trackmate file as string
    fileID = fopen(trackmate_xml, 'r');
    trackmate_data =  textscan(fileID,'%s');
    spots_start = find(cell2mat(cellfun(@(x) startsWith(x, '<AllSpot'), trackmate_data, 'un', 0))) +1;
    spots_end = find(cell2mat(cellfun(@(x) startsWith(x, '</AllSpot'), trackmate_data, 'un', 0))) -1;
    tracks_start = find(cell2mat(cellfun(@(x) startsWith(x, '<AllTracks'), trackmate_data, 'un', 0))) + 1;
    tracks_end = find(cell2mat(cellfun(@(x) startsWith(x, '</AllTracks'), trackmate_data, 'un', 0))) -1;
    

    spots_data = trackmate_data{1}(spots_start:spots_end);
    
    % inefficient but IDC
    IDs = extractFromSpots('ID', spots_data);
    Z = extractFromSpots('POSITION_Z', spots_data);
    Y = extractFromSpots('POSITION_Y', spots_data);
    X = extractFromSpots('POSITION_X', spots_data);
    frames = extractFromSpots('FRAME', spots_data);
    labels = extractFromSpots('MEDIAN_INTENSITY', spots_data);
    
    
    % same story for track_data
    
    track_data = trackmate_data{1}(tracks_start:tracks_end);
    [track_ids, track_start_lines] = extractFromSpots('TRACK_ID', track_data);

    [spot_start_id, spot_start_lines] =  extractFromSpots('SPOT_SOURCE_ID', track_data);
    [spot_end_id, spot_end_lines] =  extractFromSpots('SPOT_TARGET_ID', track_data);
    
    
    
    corresponding_track_idcs_start = discretize(find(spot_start_lines), [find(track_start_lines); numel(track_start_lines)+1]);
    corresponding_track_idcs_end = discretize(find(spot_end_lines), [find(track_start_lines); numel(track_start_lines)+1]);
    
    assert(all(corresponding_track_idcs_start == corresponding_track_idcs_end))
    
    edge_track_ids = track_ids(corresponding_track_idcs_start);
    
    assert(numel(unique(spot_start_id)) == numel(spot_start_id));
    
    track_ids_start = nan(numel(IDs), 1);
    track_ids_end = nan(numel(IDs), 1);
    
    
    [is_member_start, idx] = ismember(spot_start_id, IDs);
    track_ids_start(idx) = edge_track_ids;
    
    [is_member_end, idx] = ismember(spot_end_id, IDs);
    track_ids_end(idx) = edge_track_ids;
    
    
    sel = ~isnan(track_ids_end) &  ~isnan(track_ids_start);
    
    assert(all(track_ids_end(sel) == track_ids_start(sel)))
    
    tracks = nan(numel(IDs), 1);
    
    tracks(~isnan(track_ids_end)) = track_ids_end(~isnan(track_ids_end));
    tracks(~isnan(track_ids_start)) = track_ids_start(~isnan(track_ids_start));
    
    % random shuffiling for easier visualization
    unique_track_ids = unique(tracks(~isnan(tracks)));
    new_track_ids = unique_track_ids(randperm(length(unique_track_ids)));
    tracks(~isnan(tracks)) = new_track_ids(tracks(~isnan(tracks))+1);
    
    assert(numel(tracks) == numel(frames))
    
    segmentation_files = dir(fullfile(prediction_folder, '*.tif'));
    
    scaling_factors = [4, 1, 1];
        
    for i = 1:numel(segmentation_files)
        segmentation_file_path = fullfile( ...
            segmentation_files(i).folder, segmentation_files(i).name);
        
        labelimage = imread3D(segmentation_file_path);
        objects = conncomp(labelimage);

        frame_selection = frames == i-1 & ~isnan(tracks);
        tracks_in_frame = tracks(frame_selection);
        labels_in_frame = labels(frame_selection);
        
        [~, idcs] = ismember(labels_in_frame, 1:objects.NumObjects);
        assert(all(labels_in_frame <= objects.NumObjects));
        assert(all(labels_in_frame > 0));
        
        
        track_ids = nan(objects.NumObjects, 1);
        track_ids(idcs) = tracks_in_frame;

        objects.stats = regionprops(objects, 'BoundingBox', 'Centroid');
        
        track_ids = num2cell(track_ids);
        [objects.stats.Track_ID] = track_ids{:};
        
        output_file = fullfile(output_path, sprintf('frame%06d.vtk', i));
        fprintf('%s %d\n', output_file, isfolder(output_path));
        objects2VTK(output_file, objects, 0.2);

    end

end
