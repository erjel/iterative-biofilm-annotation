function run_huygens_parameter_scan(batch_file, input_file, prep_folder, result_folder)

	biofilmq_path_RAIMO_COMPUTER = 'C:\Users\Eric\src\bacbq\includes';
	addpath(genpath(biofilmq_path_RAIMO_COMPUTER));
    
    im = imread3D(input_file);
    im(:, :, 1) = [];
	%im = im(:, :, 1:4:end);
    
    [folder, file_name, ext] = fileparts(input_file);
	
	if ~isfolder(prep_folder)
		mkdir(prep_folder)
	end
	
	if ~isfolder(result_folder)
		mkdir(result_folder)
	end
    
    prep_file_path = fullfile(prep_folder, [file_name, ext]);
    imwrite3D(im, prep_file_path)
       
    decon_params = [];
    test_iterations = 5:5:60;
    for i = 1:numel(test_iterations)
        num_iterations = test_iterations(i);
        decon_params(i).microscope_template = 'drescher_100x_SiOil_2xlens'; % or 'drescher_100x_Oil_2xlens'
        decon_params(i).deconvolution_template = 'biofilms'; % 
        decon_params(i).input_file_path = prep_file_path;
        decon_params(i).output_dir = result_folder;
        decon_params(i).output_file_name = sprintf('%s_iter%d%s', file_name, num_iterations, ext);
        decon_params(i).dxy = 0.0592;
        decon_params(i).dz = 0.4;
        decon_params(i).numerical_aperture_objective = 1.35;
        decon_params(i).refractive_index_medium = 1.406;
        decon_params(i).excitation_wavelength = 488;
        decon_params(i).emission_wavelength = 520;
        decon_params(i).num_iterations = num_iterations;
        decon_params(i).quality_threshold = 0.001;
        decon_params(i).signal_to_noise_ratio = 20;
    end
    
    generateHuygensBatchFile_clean(batch_file, decon_params);

    