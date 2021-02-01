% function extractRegistration(registration_csv)

input_dir = "Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5";
pattern = '*_metadata.mat';
file_list = dir(fullfile(input_dir, pattern));

N = numel(file_list);
translations = zeros(N, 3);

for i = 1:N
    metadata = load(fullfile(file_list(i).folder, file_list(i).name), 'data');
    translations(i, :) = metadata.data.registration.T(end, 1:3);
end

writematrix(translations,'translations.csv') 