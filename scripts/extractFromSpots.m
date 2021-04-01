    function [results, is_id] = extractFromSpots(value_str, spots_data)
        is_id = cell2mat(cellfun(@(x) startsWith(x, value_str), spots_data, 'un', 0));
        results = cell2mat(cellfun(@(x) str2double(x(numel(value_str)+3:end-1)), spots_data(is_id), 'un', 0));
    end