function data2labelimage(output_folder, input_folder)
    biofilmq_path = fullfile(getenv('Home'), 'src', 'BiofilmQ', 'includes');
    addpath(genpath(biofilmq_path));
    
    pattern = '*_data.mat';
    
    file_list = dir(fullfile(input_folder, pattern));
    
    if ~isfolder(output_folder)
        mkdir(output_folder)
    end
    
    
    
    for i = 1:numel(file_list)
        fname = file_list(i).name;
        folder = file_list(i).folder;
        data = load(fullfile(folder, fname), ...
            'Connectivity', 'ImageSize', 'NumObjects', 'PixelIdxList');
        w = labelmatrix(data);
        
        output_name = strrep(fname, '_data.mat', '.tif');
        imwrite3D(w, fullfile(output_folder, output_name));
    end
    
    return
end