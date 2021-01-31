%function biofilmQ2trackMate(output_xml, input_folder, tifffile, pattern)

input_folder = '/home/eric/Downloads/data';
pattern = '*_data.mat';

file_list = dir(fullfile(input_folder, pattern));

tiff_stack_name = 'dummy.tif';
tiff_stack_folder = 'dummy_folder/folder';

Nx = 1024;
Ny = 1024;
Nz = 54;
Nt = 114;
dx = 0.063;
dy = 0.063;
dz = 0.4;


dt = 1.0;

spot_template = {''};


n_total = 0;
for i = 1:numel(file_list)
    spot_template{end+1} = sprintf('      <SpotsInFrame frame="%d">', i);
    stats_ = load(fullfile(file_list(i).folder, file_list(i).name), 'stats');
    stats_ = stats_.stats;
    for j = 1:numel(stats_)
        n_total = n_total + 1;
        spot_template{end+1} = [ ...
            sprintf('        <Spot ID="%d" ', n_total), ...
            sprintf('name="ID%d" ', n_total), ...
            sprintf('QUALITY="%f" ', 1), ...
            sprintf('POSITION_T="%f" ', i*dt), ...
            sprintf('MAX_INTENSITY="%d" ', 1), ...
            sprintf('FRAME="%d" ', i), ...
            sprintf('MEDIAN_INTENSITY="%f" ', 1), ...
            sprintf('VISIBILITY="%f" ', 1), ...
            sprintf('MEAN_INTENSITY="%f" ', 1), ...
            sprintf('TOTAL_INTENSITY="%f" ', 1), ...
            sprintf('ESTIMATED_DIAMETER="%f" ', 1), ...
            sprintf('RADIUS="%f" ', 1), ...
            sprintf('SNR="%f" ', 1), ...
            sprintf('POSITION_X="%f" ', 1), ...
            sprintf('POSITION_Y="%f" ', 1), ...
            sprintf('STANDARD_DEVIATION="%f" ', 1), ...
            sprintf('CONTRAST="%f" ', 1), ...
            sprintf('MANUAL_COLOR="%d" ', 1), ...
            sprintf('MIN_INTENSITY="%f" ', 1), ...
            sprintf('POSITION_Z="%f" />', 1)];
    end
    spot_template{end+1} = sprintf('      </SpotsInFrame>');
    n_total_new = n_total+numel(stats_);
    ids = num2cell(n_total+1:n_total_new);
    
    frame = num2cell(ones(size(ids)) *i);
    
    [stats_.id] = ids{:};
    [stats_.frame] = frame{:};
    if i == 1
        stats = stats_;
    else
        stats = [stats; stats_]; % is a waste of resources ...
    end
    
    n_total = n_total_new;
end

spot_template{1} = sprintf('    <AllSpots nspots="%d">', n_total)    ;
spot_template{end+1} = '    </AllSpots>';
spot_template = cellfun(@(line) [line, newline], spot_template, 'un', 0);

spot_template = sprintf('%s', spot_template{:});
% write settings:

image_data_settings = [ ...
    sprintf('    <ImageData filename="%s" ', tiff_stack_name), ...
    sprintf('folder="%s" ', tiff_stack_name), ...
    sprintf('width="%d" ', Nx), ...
    sprintf('height="%d" ', Ny), ...
    sprintf('nslices="%d" ', Nz), ...
    sprintf('nframes="%d" ', Nt), ...
    sprintf('pixelwidth="%f" ', dx), ...
    sprintf('pixelheight="%f" ', dy), ...
    sprintf('voxeldepth="%f" ', dz), ...
    sprintf('timeinterval="%f" />', dt)];

basic_settings = [ ...
    sprintf('    <BasicSettings xstart="0" xend="%d" ', Nx-1), ...
    sprintf('ystart="0" yend="%d" ', Ny-1), ...
    sprintf('zstart="0" zend="%d" ', Nz-1), ...
    sprintf('tstart="0" tend="%d" />', Nt-1)];


fname = '../resources/template_TrackMate_MATLAB.xml';

boilerplate_text = fileread(fname);
boilerplate_text = splitlines(boilerplate_text);
boilerplate_text = cellfun(@(line) [line, newline], boilerplate_text, 'un', 0);
boilerplate_text = sprintf('%s', boilerplate_text{:});
xml_file  = sprintf(boilerplate_text, ...
    spot_template, image_data_settings, basic_settings);


mid = fopen('debug.xml', 'w');
fprintf(mid,'%s',xml_file);
fclose(mid);



%end