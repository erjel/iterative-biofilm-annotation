addpath(genpath('D:\Users\Eric\src\bacbq')) % commit c2d381d2a7b7ce396c84f10d10745dd3585ec48b 

filelist = dir('input_data\*.tif');

is_labelimage = num2cell(cellfun(@(x) endsWith(x, 'L.tif'), {filelist.name}))
[filelist.is_labelimage] = is_labelimage{:};

associated_biofilm = {2, 2, 1, 1, 4, 3, 3, 4, 5, 5};
[filelist.biofilm_id] = associated_biofilm{:};

% have to f*cking interpolate
params = [];
params.scaleUp = false;
silent = false;
dxy = 61;
dz = 100;

scalebar_um = 10; 

visual_inspection = false
if visual_inspection
for i = 1:numel(filelist)
    if ~filelist(i).is_labelimage
        im = imread3D(fullfile(filelist(i).folder, filelist(i).name));
        
        im = zInterpolation(im, dxy, dz, params, silent);
       
        zSlicer(im, parula)
    end
end
end

% manually selected slices:
x = {537, 537, 513, 513, 511, 483, 483, 511, 497, 497};
y = {301, 301, 345, 345, 335, 372, 372, 335, 404, 404};
z = {27, 27, 42,42,26, 34,34 26, 36, 36};

[filelist.x] = x{:};
[filelist.y] = y{:};
[filelist.z] = z{:};

output_folder = 'sliced_views';

if ~isfolder(output_folder)
    mkdir(output_folder)
end

unique_biofilms = unique([filelist.biofilm_id]);

for i = 1:numel(unique_biofilms)
    biofilm_id = unique_biofilms(i);
    j = find(~[filelist.is_labelimage] & [filelist.biofilm_id] == biofilm_id);
    
    im = imread3D(fullfile(filelist(j).folder, filelist(j).name));
    
    x = filelist(j).y;
    y = filelist(j).x;
    z = filelist(j).z;
    
    im = zInterpolation(im, dxy, dz, params, silent);
    
    im_xy = squeeze(im(:, :, z));
    im_xz = squeeze(im(:, y, :));
    im_yz = squeeze(im(x, :, :));
    
    imgs = {im_xy, im_xz, im_yz};

    

    % clim = [0, prctile([im_xy(:); im_xz(:); im_yz(:)], 98)];

    % xy
    f = figure;
    ax = axes(f);
    hold(ax, 'on')
    % imagesc(ax, imgs_signal{1});
    % visboundaries(ax, imgs{1});
    imagesc(ax, imgs{1});
    plot(ax, [0, 1024], [x, x], 'r', 'linewidth', 3);
    plot(ax, [y, y],  [0, 1024], 'g', 'linewidth', 3);

    scalebar_px = scalebar_um / 0.061;
    plot(ax, [950-scalebar_px, 950], [20, 20], 'w', 'LineWidth', 5);

    ax.DataAspectRatio = [1, 1, 1];
    ax.YLim = ([0, size(imgs{1}, 1)]);
    ax.XLim = ([0, size(imgs{1}, 2)]);
    cmap = gray(256);
    cmap(:, 1) = 0;
    cmap(:, 3) = 0;
    colormap(cmap);
    colorbar;
    ax.ColorScale = 'log';

    clim = ax.CLim;

    exportgraphics(f, fullfile(output_folder, sprintf('biofilm_%d_xy_slice.eps', biofilm_id)));

    f = figure;
    ax = axes(f);
    hold(ax, 'on');
    % imagesc(ax, imgs_signal{2});
    % visboundaries(ax, imgs{2});
    imagesc(ax, imgs{2});
    plot(ax, [0, size(imgs{2}, 2)], [x, x], 'r', 'linewidth', 3);
    plot(ax, [z, z], [0, size(imgs{2}, 1)], 'b', 'linewidth', 3);


    ax.DataAspectRatio = [1, 1, 1];
    ax.YLim = ([0, size(imgs{2}, 1)]);
    ax.XLim = ([0, size(imgs{2}, 2)]);
    cmap = gray(256);
    cmap(:, 1) = 0;
    cmap(:, 3) = 0;
    colormap(cmap);
    colorbar;
    ax.CLim = clim;
    ax.ColorScale = 'log';
    exportgraphics(f, fullfile(output_folder, sprintf('biofilm_%d_xz_slice.eps', biofilm_id)));


    f = figure;
    ax = axes(f);
    hold(ax, 'on');
    im_yz = imgs{3}';
    % im_yz_signal = imgs_signal{3}';
    imagesc(ax, im_yz);
    % visboundaries(ax, im_yz);

    plot(ax, [0, size(im_yz, 2)], [z, z], 'b', 'linewidth', 3);
    plot(ax, [y, y], [0, size(im_yz, 1)], 'g', 'linewidth', 3);
    ax.DataAspectRatio = [1, 1, 1];
    ax.YLim = ([0, size(im_yz, 1)]);
    ax.XLim = ([0, size(im_yz, 2)]);
    cmap = gray(256);
    cmap(:, 1) = 0;
    cmap(:, 3) = 0;
    colormap(cmap);
    colorbar;
    ax.CLim = clim;
    ax.ColorScale = 'log';
    exportgraphics(f, fullfile(output_folder, sprintf('biofilm_%d_yz_slice.eps', biofilm_id)));
end

