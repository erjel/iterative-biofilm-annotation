addpath(genpath('D:/Users/Eric/src/bacbq'))

gt_paths = {'D:\Users\Eric\Documents\phd-thesis\data\chapter_single_cell_segmentation\full_semimanual-huy\test\masks\im0.tif', ...
    'D:\Users\Eric\Documents\phd-thesis\data\chapter_single_cell_segmentation\fig_accuracy_measurements\full_stacks_huy\masks_intp\Pos1_ch1_frame000001_Nz300.tif'};

fn_paths = {'stardist_fn.tif', ...
    'biofilmQ_fn_v2.tif'};

fp_paths = {'stardist_fp.tif', ...
    'biofilmQ_fp_v2.tif'};


result_paths = {'D:\Users\Eric\Documents\phd-thesis\data\chapter_deep_learning_segmentation\predictions\stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep5\datasets\full_semimanual-raw\test\images\im0.tif', ...
    'D:\Users\Eric\Documents\phd-thesis\data\chapter_single_cell_segmentation\fig_accuracy_measurements\full_stacks_huy\predictions\data_seeded_watershed\Pos1_ch1_frame000001_Nz300.tif'};

intensity = {'D:\Users\Eric\Documents\phd-thesis\data\chapter_single_cell_segmentation\fig_accuracy_measurements\full_stacks_raw\images\Pos1_ch1_frame000001_Nz300.tif', ...
    'D:\Users\Eric\Documents\phd-thesis\data\chapter_single_cell_segmentation\fig_accuracy_measurements\full_stacks_raw\images\Pos1_ch1_frame000001_Nz300.tif'};

algorithms = {'stardist', 'BiofilmQ'};

error_types = {'false negative', 'false_positive'};

silent = true;
params = struct('scaleUp', false);
dxy = 0.061; % in um
dz = 0.100; % in um

scalebar_um = 2;

rng(42);
offset = 50;

output_folder = 'outputs';
if ~isfolder(output_folder)
    mkdir(output_folder);
end



%for algorithm_id = 1:numel(algorithms)
for algorithm_id = 1
    algorithm = algorithms{algorithm_id};
    
    for error_id = 1:numel(error_types)
        error_type = error_types{error_id};
        
        if strcmp(error_type, 'false negative')
            error_volume = imread3D(fn_paths{algorithm_id});
        else
            error_volume = imread3D(fp_paths{algorithm_id});
        end
        
        gt = imread3D(gt_paths{algorithm_id});
        pred = imread3D(result_paths{algorithm_id});
        int_img = imread3D(intensity{algorithm_id});
        int_img(:, :, 1) = [];
        int_img = zInterpolation(int_img, dxy, dz, params, silent);
        
        % corrections
        if strcmp(algorithm, 'stardist')
            gt(:, :, 1) = [];
            gt =  zInterpolation_nearest(gt, dxy, dz, params, silent);
            
            pred =  zInterpolation_nearest(pred, dxy, dz, params, silent);
            error_volume = zInterpolation_nearest(error_volume, dxy, dz, params, silent);
        end
        
        if strcmp(error_type, 'false negative')
            gt_ids = unique(gt .* uint16(error_volume == 1));
            selection = randi(numel(gt_ids),5, 1);
            cc = conncomp(gt);
        else
            pred_ids = unique(pred .* uint16(error_volume == 1));
            selection = randi(numel(pred_ids),5, 1);
            cc = conncomp(pred);
        end
        
        stats = regionprops(cc);
        
        for j_ = 1:numel(selection)
            
            if strcmp(error_type, 'false negative')
                j = gt_ids(selection(j_));
            else
                 j = pred_ids(selection(j_));
            end
            center = stats(j).Centroid;
            
            center_start = round(center - offset);
            center_start = arrayfun(@(x) max([x, 1]), center_start);
            
            center_end = round(center + offset);
            center_end = arrayfun(@(x, y) min([x, y]), [cc.ImageSize(2), cc.ImageSize(1), cc.ImageSize(3)], center_end);
            
            x = round(center(1));
            y = round(center(2));
            z = round(center(3));
            
            imgs_pred = {...
                squeeze(pred(center_start(2):center_end(2), center_start(1):center_end(1), z)), ...
                squeeze(pred(center_start(2):center_end(2), x, center_start(3):center_end(3))), ...
                squeeze(pred(y, center_start(1):center_end(1), center_start(3):center_end(3)))'};
            
            imgs_gt = { ...
                squeeze(gt(center_start(2):center_end(2), center_start(1):center_end(1), z)), ...
                squeeze(gt(center_start(2):center_end(2), x, center_start(3):center_end(3))), ...
                squeeze(gt(y, center_start(1):center_end(1), center_start(3):center_end(3)))'};
            
            imgs_int = { ...
                squeeze(int_img(center_start(2):center_end(2), center_start(1):center_end(1), z)), ...
                squeeze(int_img(center_start(2):center_end(2), x, center_start(3):center_end(3))), ...
                squeeze(int_img(y, center_start(1):center_end(1), center_start(3):center_end(3)))'};
            
            % clim = [0, prctile([im_xy(:); im_xz(:); im_yz(:)], 98)];
            
            % xy
            f = figure;
            ax = axes(f, 'OuterPosition', [0.0 0.5 0.5 0.5]);
            title(error_type)
            hold(ax, 'on')
            % imagesc(ax, imgs_signal{1});
            % visboundaries(ax, imgs{1});
            
            imagesc(ax, imgs_int{1});
            gt_vals = unique(imgs_gt{1});
            for k = 2:numel(gt_vals)
                visboundaries(ax, imgs_gt{1} == gt_vals(k), 'Color', 'b');
            end
            
            pred_vals = unique(imgs_pred{1});
            for k = 2:numel(pred_vals)
                visboundaries(ax, imgs_pred{1} == pred_vals(k), 'Color', 'r');
            end
            scatter(ax, center(2)-center_start(2), center(1)-center_start(1), 'y', 'filled')
            % plot(ax, [0, 1024], [x, x], 'r', 'linewidth', 3);
            % plot(ax, [y, y],  [0, 1024], 'g', 'linewidth', 3);
            
            scalebar_px = scalebar_um / 0.061;
            plot(ax, [0.95*ax.YLim(2)-scalebar_px, 0.95*ax.YLim(2)], [0.05*ax.YLim(1), 0.05*ax.YLim(1)], 'w', 'LineWidth', 5);
            
            ax.DataAspectRatio = [1, 1, 1];
            ax.YLim = ([0, size(imgs_int{1}, 1)]);
            ax.XLim = ([0, size(imgs_int{1}, 2)]);
            cmap = gray(256);
%             cmap(:, 1) = 0;
%             cmap(:, 3) = 0;
            colormap(cmap);
            
            clim = ax.CLim;
            
            
            
            
            ax = axes(f, 'OuterPosition', [0.5 0.5, 0.5, 0.5]);
            title(ax, algorithm)
            hold(ax, 'on');
            
            imagesc(ax, imgs_int{2});
            
            gt_vals = unique(imgs_gt{2});
            for k = 2:numel(gt_vals)
                visboundaries(ax, imgs_gt{2} == gt_vals(k), 'Color', 'b');
            end
            
            pred_vals = unique(imgs_pred{2});
            for k = 2:numel(pred_vals)
                visboundaries(ax, imgs_pred{2} == pred_vals(k), 'Color', 'r');
            end
            
            scatter(ax,  center(3)-center_start(3), center(2)-center_start(2), 'y', 'filled')
            % plot(ax, [0, size(imgs{2}, 2)], [x, x], 'r', 'linewidth', 3);
            % plot(ax, [z, z], [0, size(imgs{2}, 1)], 'b', 'linewidth', 3);
            
            
            ax.DataAspectRatio = [1, 1, 1];
            ax.YLim = ([0, size(imgs_int{2}, 1)]);
            ax.XLim = ([0, size(imgs_int{2}, 2)]);
            cmap = gray(256);
%             cmap(:, 1) = 0;
%             cmap(:, 3) = 0;
            colormap(cmap);
            ax.CLim = clim;
            %exportgraphics(f, fullfile(output_folder, sprintf('biofilm_%d_xz_slice.eps', biofilm_id)));
            
            
            pos_xy = [0.1 0.3 0.3 0.3];
            pos_yz = [0.5 0.15, 0.5, 0.7];
            
            ax = axes(f, 'OuterPosition', [0.0, 0.0, 0.5, 0.5]);
            
            
            hold(ax, 'on');
            
            imagesc(ax, imgs_int{3}(:, end:-1:1));
            
            gt_vals = unique(imgs_gt{3});
            for k = 2:numel(gt_vals)
                visboundaries(ax, imgs_gt{3}(:, end:-1:1) == gt_vals(k), 'Color', 'b');
            end
            
            pred_vals = unique(imgs_pred{3});
            for k = 2:numel(pred_vals)
                visboundaries(ax, imgs_pred{3}(:, end:-1:1) == pred_vals(k), 'Color', 'r');
            end
            scatter(ax, center(1)-center_start(1), center(3)-center_start(3), 'y', 'filled')
            
            % plot(ax, [0, size(im_yz, 2)], [z, z], 'b', 'linewidth', 3);
            % plot(ax, [y, y], [0, size(im_yz, 1)], 'g', 'linewidth', 3);
            ax.DataAspectRatio = [1, 1, 1];
            %     ax.YLim = ([0, size(im_yz, 1)]);
            %     ax.XLim = ([0, size(im_yz, 2)]);
            cmap = gray(256);
%             cmap(:, 1) = 0;
%             cmap(:, 3) = 0;
            colormap(cmap);
            ax.CLim = clim;
            %exportgraphics(f, fullfile(output_folder, sprintf('biofilm_%d_yz_slice.eps', biofilm_id)));
            
            ax = axes(f, 'OuterPosition', [0.5, 0.0, 0.5, 0.5]);
            
            % calculate IoU value
            
            if strcmp(error_type, 'false negative')
                overlap_ids = unique(pred(cc.PixelIdxList{j}));
            else
                overlap_ids = unique(gt(cc.PixelIdxList{j}));
            end
            ious = {};
            for l = 1:numel(overlap_ids)
                overlap_id = overlap_ids(l);
                if overlap_id ~= 0
                    if strcmp(error_type, 'false negative')
                        ious{end+1} =  sum(gt(pred == overlap_id) == j) / sum((gt == j) | (pred == overlap_id), 'all');
                    else
                        ious{end+1} =  sum(pred(gt == overlap_id) == j) / sum((pred == j) | (gt == overlap_id), 'all');
                    end
                end
            end
            
            iou = max(cell2mat(ious));
            fprintf('%f\n', iou);
            title(ax, sprintf('IoU = %f', max(cell2mat(ious))));
            
            colorbar()
            ax.DataAspectRatio = [1, 1, 1];
            ax.YLim = ([0, size(imgs_int{3}, 1)]);
            ax.XLim = ([0, size(imgs_int{3}, 2)]);
            cmap = gray(256);
%             cmap(:, 1) = 0;
%             cmap(:, 3) = 0;
            colormap(cmap);
            ax.CLim = clim;
            
            
            exportgraphics(f, fullfile(output_folder, sprintf('error_plot_%s_%s_%d.eps', algorithm, error_type, j_)));
            close all;
        end
        
        
    end
end