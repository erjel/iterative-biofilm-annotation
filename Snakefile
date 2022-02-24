# snakemake -j --profile cobra --use-conda
configfile: "config.yml"

include: r"workflows/rules/snakefile_care"
include: "workflows/rules/stardist.smk"
include: "workflows/rules/cellpose.smk"
include: "workflows/rules/stardist_merge.smk"
include: "workflows/rules/biofilmq.smk"
include: "workflows/rules/trackmate.smk"
include: "workflows/rules/analysis.smk"
include: "workflows/rules/figures.smk"

from pathlib import Path


rule all:
    input:
        #'models/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep5'
        # TODO(erjel): Here, I use the "care" enhanced dataset instead of "raw" ...
        # TODO(erjel): stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep1 was not trained in this pipeline .. replace!
        #'interim_data/predictions/care/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep1/.chkpnt',
        #'interim_data/trackmate_stacks/care.tif'
        'interim_data/tracking/care_model_eva-v1-dz400-care_rep1.xml',
        # TODO(erjel): Now open trackmate and preform the tracking ...
        # TODO: "tracks/{data}_model_{model}.csv",
        # TODO(gatoniel): Figure out how to reduce the memory footprint
        #'interim_data/predictions/care_merge/eva-v1-dz400-care_rep1_merge/.chkpnt',
        ## Try to recreate fig3a:
        #'interim_data/predictions/full_semimanual-raw/test/images/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep2'
        #'interim_data/predictions/full_semimanual-raw/test/images/horovod_cellpose_patches-semimanual-raw-64x128x128_prc100_bs8_lr0.00625_wd0.00001_mt0.7_sge_rep1'
        'interim_data/predictions/full_semimanual-raw/test/images/data_seeded_watershed',
        #'figures/fig3a'

        #expand(r"data\interim\vtk\frame_{frame_number}.vtk", 
        #    frame_number = glob_wildcards(r"predictions\{label_1}_frame{frame_number}_{label_2}.tif")[1]
        #),
        #r"reports\figures\segmentation_rendered.mp4",
        #r"reports\figures\segmentation_rendered.gif",
        ##expand(r"data\interim\care\{label_1}_ch3_{label_2}.tif",
        ##   label_1 = glob_wildcards(r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\{label_1}_ch1_{label_2}.tif")[0][:5],
        ##   label_2 = glob_wildcards(r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\{label_1}_ch1_{label_2}.tif")[1][:5]),
        #r"data\interim\care",
        #r"..\2021_Iterative_Biofilm_Annotation_bk\datasets\.eva-v1-dz400-care.chkpnt",
        #r"models\eva-v1-dz400-care_rep1",
        #r"data\interim\predictions\care\eva-v1-dz400-care_rep1\.chkpnt",
        #expand(r"data\interim\vtk\care\eva-v1-dz400-care_rep1\frame_{frame_number}.vtk", 
        #    frame_number = glob_wildcards(r"data\interim\predictions\care\eva-v1-dz400-care_rep1\{label_1}_frame{frame_number}_{label_2}.tif")[1]
        #),
        #r"reports\figures\care\eva-v1-dz400-care_rep1_segmentation_rendered.mp4",
        #r"data\processed\tracks\care_model_eva-v1-dz400-care_rep1.csv",
        #r"data\processed\tracks\care_model_eva-v1-dz400-care_rep1_growthrate.csv",
        #r"data\interim\tracking\care_model_eva-v1-dz400-care_rep1.xml",
        #r'reports\figures\care\eva-v1-dz400-care_rep1_single_cell_growthrate.png',
        #r'reports\figures\care\eva-v1-dz400-care_rep1_growthrate_heatmap.png',
        #r'reports\figures\care\BiofilmQ_single_cell_growthrate.png',
        #r'reports\figures\care\BiofilmQ_growthrate_heatmap.png',
        #expand(r"data\interim\vtk\care\BiofilmQ\frame_{frame_number}.vtk", 
        #    frame_number = glob_wildcards(r"data\interim\predictions\care\BiofilmQ\{label_1}_frame{frame_number}_{label_2}.tif")[1]
        #),
        #'data/interim/metadata/raw/z_standard_deviation.csv',
        #'data/interim/training_sets/CARE_2D/raw_ch1_ch2/2021-02-03.npz',
        #'models/care/2D_raw_ch1_ch2_2021-02-03_rep1',
        #'data/interim/training_sets/CARE_3D/raw_ch1_ch2/2021-02-03.npz',
        #'models/care/3D_raw_ch1_ch2_2021-02-03_rep1',
        #'models/n2v/3D_raw_ch1_rep1',
        #expand('data/interim/predictions/raw/n2v/n2v_3D_small/{label_1}_ch1_{label_2}.tif', 
        #    label_1 = glob_wildcards(r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\{label_1}_ch1_{label_2}.tif")[0],
        #    label_2 = glob_wildcards(r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\{label_1}_ch1_{label_2}.tif")[1],
        #),
        #'data/interim/predictions/raw/care/3D_raw_ch1_ch2_2021-02-03_rep1/.chkpnt',
        #'data/interim/predictions/raw/care/2D_raw_ch1_ch2_2021-02-03_rep1/.chkpnt',
        #'models/care/3D_raw_ch2_ch1_2020-01-05_rep1', # Uses the training data in "Y:\Eva\CARE\Created_Trainingdata\2020-01-05_ch2_ch1_testsplit.npz"
        #'models/care/3D_raw_ch2_ch1_2020-01-05_rep2',
        #'models/care/3D_raw_ch2_ch1_2020-01-05_rep3',
        #'models/care/3D_raw_ch2_ch1_2020-01-05_rep4',
        #'models/care/3D_raw_ch2_ch1_2020-01-05_rep5',
        ##'G:/batch_decon/data/interim/huygens_parameterscan/huygens_batch_file_rep1.hgsb',
        ##'data/interim/huygens_parameterscan',
        #'data/processed/tracks/care_model_eva-v1-dz400-care_rep1_vtk',
        #'data/processed/tracks/care_model_eva-v1-dz400-care_rep1_tif',

localrules:
    bronto_download_dataset

rule bronto_download_dataset:
    output:
        directory('training_data/{dataset}'),
    input:
        'training_data/.symlink',
    shell:
        'scp -r bronto:/volume1/bronto/Eric/2021_Iterative_Biofilm_Annotation_bk/datasets/{wildcards.dataset} ./training_data/'

localrules:
    create_symlinks

rule create_symlinks:
    output:
        touch('{directory}/.symlink')
    params:
        target = lambda wc: config["symlinks"][wc.directory]
    shell:
        "ln -s {params.target} {wildcards.directory}"

        
rule copy_huygens:
    output:
        directory('data/interim/huygens_parameterscan'),
    params:
        'G:/batch_decon/data/interim/huygens_parameterscan'
    threads:
        1
    shell:
        r"XCOPY {params} {output} /s /i"
        
rule huygens_parameterscan:
    output:
        'G:/batch_decon/data/interim/huygens_parameterscan/huygens_batch_file_rep1.hgsb',
    input:
        #r"Y:\Eva\CARE\08.12.20_19h\plasmid-100nm-19h\plasmid-100nm-19h_pos5_ch1_frame000001_Nz271.tif",
        r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame000083_Nz54.tif",
    params:
        prep_folder = 'G:/batch_decon/data/interim/huygens_parameterscan/prep_folder',
        result_folder = 'G:/batch_decon/data/interim/huygens_parameterscan/result_folder',
    threads:
        1
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); run_huygens_parameter_scan('{output}',  '{input}', '{params.prep_folder}', '{params.result_folder}')" """

rule predict_n2v:
    output:
        'data/interim/predictions/raw/n2v/{model}/{tiff}.tif',
    params:
        model_path = r"\models\n2v\{model}",
        input_folder = r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5',
    resources:
        nvidia_gpu=1
    conda:
        'envs/n2v.yml'
    shell:
        r'python scripts/n2v_prediction.py {output} "{params.input_folder}\{wildcards.tiff}.tif" {params.model_path}' +\
            ' --has_overview'

rule train_n2v:
    output:
        directory('models/n2v/3D_{data}_{ch}_rep{rep}')
    params:
        input_folder=r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5',
    resources:
        nvidia_gpu=1
    conda:
        'envs/n2v.yml'
    shell:
        r'python scripts/n2v_training.py {output} "{params.input_folder}" --seed {output}'

rule calc_coverslip_slice:
    output:
        std_csv = 'data/interim/metadata/{data}/z_standard_deviation.csv',
        cover_slip_slice_csv = 'data/interim/metadata/{data}/cover_slip_slice.csv',
    input:
        basepaths = ['Y:/Eva/CARE/08.12.20_14h', 'Y:/Eva/CARE/08.12.20_19h', 'Y:/Eva/CARE/08.12.20_24h', 'Y:/Eva/CARE/09.12.20_14h']
    conda:
        'envs/calc.yml'
    shell:
        'python scripts/calc_coverslip_slice.py {output.std_csv} {output.cover_slip_slice_csv} {input.basepaths}'

#TODO(erjel): Are the following rules unused?
rule biofilmQData2Labelimages:
    output:
        directory(r'Y:\Eric\prediction_test\data\interim\predictions\{data}\BiofilmQ'),
    params:
        input_folder = r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\data',
    threads:
        1
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); data2labelimage('{output}',  '{params.trans}', '{output.input_folder}')" """

rule create_vtk_from_labelfile_test:
    output:
        r"data\interim\vtk\frame_{frame_number}.vtk"
    input:
        r"predictions\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame{frame_number}_Nz54P.tif"
    threads:
        2
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tif2vtk('{output}', '{input}')" """
        
rule create_vtk_from_labelfile_general:
    output:
        r"data\interim\vtk\{data}\{model}\frame_{frame_number}.vtk"
    input:
        r"data\interim\predictions\{data}\{model}\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame{frame_number}_Nz54.tif"
    threads:
        2
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tif2vtk('{output}', '{input}')" """
        
        
rule create_vtk_from_labelfile_general_ch2:
    output:
        r"data\interim\vtk\{data}\{model}\frame_{frame_number}.vtk"
    input:
        r"data\interim\predictions\{data}\{model}\kdv1502R_5L_30ms_300gain002_pos5_ch2_frame{frame_number}_Nz54.tif"
    threads:
        2
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tif2vtk('{output}', '{input}')" """

        
rule create_rendering_test:
    output:
        r"reports\figures\segmentation_rendered.mp4"
    shell:
        r"C:\ffmpeg-3.4.2-win64-static\bin\ffmpeg.exe -pattern_type sequence -start_number 0 -framerate 15 -i data\interim\vtk_rendering\simple.%04d.png  -c:v libx264 reports\figures\segmentation_rendered.mp4"
        
rule create_rendering_general:
    output:
        r"reports\figures\{data}\{model}_segmentation_rendered.mp4",
    conda:
        r"envs\ffmpeg.yml"
    shell:
        r"ffmpeg.exe -pattern_type sequence -start_number 0 -framerate 15 -i data\interim\vtk_rendering\{wildcards.data}\{wildcards.model}\frame.%04d.png  -c:v libx264 reports\figures\{wildcards.data}\{wildcards.model}_segmentation_rendered.mp4"
        
        

rule mp4_to_gif:
    output:
        r"reports\figures\{filename}.gif"
    input:
        r"reports\figures\{filename}.mp4"
    conda:
        r"envs\ffmpeg.yml"
    shell:
        r"ffmpeg -i {input} -f gif {output}"
        