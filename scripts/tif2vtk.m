function tif2vtk(output_path, input_path)

    biofilmq_path = fullfile(getenv('Home'), 'src', 'BiofilmQ', 'includes');
    addpath(genpath(biofilmq_path));
	
	biofilmq_path_CPU_SERVER = 'D:\Eric\bacbq';
	addpath(genpath(biofilmq_path_CPU_SERVER));
	
	biofilmq_path_GPU_SERVER = 'C:\Users\Eric\src\biofilmq\includes';
	addpath(genpath(biofilmq_path_GPU_SERVER));
    
    [folder, ~] = fileparts(output_path);
    
    if ~isfolder(folder)
        fprintf('Create new output folder: %s\n', folder);
        mkdir(folder);
    end
    
    assert(isfile(input_path));
    fprintf('Read: %s\n', input_path);
    volume = imread3D(input_path);
    
    
    fprintf('Calculate connected components\n');
    objects = conncomp(volume);

    fprintf('Create vtk file\n')
    objects2VTK(output_path, objects, 0.2);
    
end