
################
# Growth rates #
################

ruleorder: tracks2growthrateBiofilmQ > tracks2growthrate

#TODO(erjel): Is this rule still needed?
rule tracks2growthrateBiofilmQ:
    output:
        r"data\processed\tracks\{data}_model_BiofilmQ_growthrate.csv",
    input:
        tracks_csv = r"data\processed\tracks\{data}_model_BiofilmQ.csv",
        prediction_folder = r"data\interim\predictions\{data}\BiofilmQ",
        translations = r"data\interim\tracking\{data}_model_BiofilmQ_translations.csv",
        crop = r"data\interim\tracking\{data}_model_BiofilmQ_crop_offsets.csv",
    threads:
        1
    conda:
        "../envs/calc.yml"
    shell:
        r"python scripts\calc_growthrate.py {output} {input.tracks_csv} {input.prediction_folder} " +
        r"--transl_csv {input.translations} --crop_csv {input.crop}"
        

rule tracks2growthrate:
    output:
        r"data\processed\tracks\{data}_model_{model}_growthrate.csv",
    input:
        tracks_csv = r"data\processed\tracks\{data}_model_{model}.csv",
        prediction_folder = r"data\interim\predictions\{data}\{model}",
    wildcard_constraints:
        model="^(?!BiofilmQ$)"
    threads:
        1
    conda:
        "../envs/calc.yml"
    shell:
        r"python scripts\calc_growthrate.py {output} {input.tracks_csv} {input.prediction_folder}"

###############################
# Model prediction accuracies #
###############################

rule calc_accuracies_biofilmq_full_stack:
    output:
        csv_file = "accuracies/data_{biofilmq_setting}/full_stacks_huy.csv",
    input:
        pred_path= "interim_data/predictions/full_stacks_huy/data_{biofilmq_setting}",
    params:
        # TODO(erjel): Where does this come from?
        gt_path = "data_BiofilmQ/full_stacks_huy/masks_intp",
    resources:
        partition = 'express',
        time="00:05:00",
        mem='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    conda:
        "../envs/stardist.yml"
    shell:
        "python iterative_biofilm_annotation/analysis/calc_accuracy_verbose.py" + \
        " {output} {input.pred_path} {params.gt_path}" + \
        " --pattern *Nz300.tif"

rule calc_accuracies:
    output:
        csv_file = "accuracies/{modelname}/{datasetname}.csv"
    input:
        pred_path="interim_data/predictions/{datasetname}/test/images/{modelname}",
        gt_path="training_data/{datasetname}/test/masks"
    threads:
        4 
    resources:
        time="00:05:00",
        mem_mb='16G',
    conda:
        "../envs/stardist.yml"
    shell:
        "python iterative_biofilm_annotation/analysis/calc_accuracy_verbose.py {output.csv_file} {input.pred_path} {input.gt_path}"


localrules: create_semi_manual_prediction
rule create_semi_manual_prediction:
    output:
        directory('interim_data/predictions/manual_raw_v3/test/images/semi_manual'),
    input:
        "training_data/full_semimanual-huy/test/masks/im0.tif",
    conda:
        "../envs/calc.yml",
    shell:
        "python iterative_biofilm_annotation/analysis/crop_biofilmq.py {output} {input}"

rule downsample_biofilmq_prediction:
    output:
        touch('.checkpoints/interim_data/predictions/full_stacks_huy/data_{biofilmq_setting}-downsampled/Pos1_ch1_frame000001_Nz300.tif.chkpt'),
        output_tif = 'interim_data/predictions/full_stacks_huy/data_{biofilmq_setting}-downsampled/Pos1_ch1_frame000001_Nz300.tif',
    params:
        input_tif = 'interim_data/predictions/full_stacks_huy/data_{biofilmq_setting}/Pos1_ch1_frame000001_Nz300.tif',
        input_dz = 61, # nm
        output_dz = 100, # nm
    conda:
        "../envs/stardist.yml",
    resources:
        partition = 'express',
        time="00:05:00",
        mem='8G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=4,
    shell:
        "python iterative_biofilm_annotation/analysis/downsample_biofilmq_prediction.py" + \
        " {output.output_tif}" + \
        " {params.input_tif}" + \
        " {params.output_dz}" + \
        " {params.input_dz}"


rule create_fn_fp_tifs:
    output:
        output_fn_tif = 'interim_data/fn_fp_visualization/{seg_type}_fn.tif',
        output_fp_tif = 'interim_data/fn_fp_visualization/{seg_type}_fp.tif',
    wildcard_constraints:
        seg_type = 'biofilmq|stardist|stardistMerge',
    input:
        # Note(erjel): I use a checkpoint here to not interfer with the standard prediction rules
        downsample_checkpoint = '.checkpoints/interim_data/predictions/full_stacks_huy/data_seeded_watershed-downsampled/Pos1_ch1_frame000001_Nz300.tif.chkpt'
    params:
        gt_path_mask = 'training_data/full_semimanual-raw/test/masks/im0.tif',
        # TODO(erjel): Include this in the overall workflow
        prediction= lambda wc: {
            'biofilmq': 'interim_data/predictions/full_stacks_huy/data_seeded_watershed-downsampled/Pos1_ch1_frame000001_Nz300.tif',
            #TODO(erjel): How to use a larger sample size than N=1?
            'stardist': 'interim_data/predictions/full_semimanual-raw/test/images/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep5/im0.tif',
            'stardistMerge': 'interim_data/predictions/full_semimanual-raw/test/images/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep5_merge/im0.tif',
        }[wc.seg_type],
        remove_pred_slice = lambda wc: {
            'biofilmq': 'False',
            'stardist': 'True',
            'stardistMerge': 'False',
        }[wc.seg_type],
    resources:
        partition = 'express',
        time="00:05:00",
        mem='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    conda:
        "../envs/stardist.yml"
    shell:
        "python iterative_biofilm_annotation/analysis/create_fn_fp_tifs.py" + \
        " {output.output_fn_tif}" + \
        " {output.output_fp_tif}" + \
        " {params.prediction}" + \
        " {params.gt_path_mask}" + \
        " {params.remove_pred_slice}"

rule convert_tif2vtk_fn:
    output:
        vtk = 'interim_data/fn_fp_visualization/{filename}_fn.vtk',
    input:
        biofilmq_includes = 'external/BiofilmQ',
        tif = 'interim_data/fn_fp_visualization/{filename}_fn.tif',
    params:
        gt_path_mask = 'training_data/full_semimanual-raw/test/masks/im0.tif',
        x_cut = 512,
        y_cut = 512,
        z_cut = 25,
    envmodules:
        'matlab/R2019b',
    resources:
        partition = 'express',
        time="00:10:00",
        mem='32G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('iterative_biofilm_annotation/analysis')); """ + \
        """ tif2vtk('{output.vtk}', '{input.tif}', '{params.gt_path_mask}', {params.x_cut}, {params.y_cut}, {params.z_cut}  )" """

rule convert_tif2vtk_fp:
    output:
        vtk = 'interim_data/fn_fp_visualization/{filename}_fp.vtk',
    input:
        biofilmq_library = 'external/BiofilmQ',
        tif = 'interim_data/fn_fp_visualization/{filename}_fp.tif',
    params:
        prediction= lambda wc: {
            'biofilmq': 'interim_data/predictions/full_stacks_huy/data_seeded_watershed-downsampled/Pos1_ch1_frame000001_Nz300.tif',
            'stardist': 'interim_data/predictions/full_semimanual-raw/test/images/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep5/im0.tif',
            'stardistMerge': 'interim_data/predictions/full_semimanual-raw/test/images/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep5_merge/im0.tif',
        }[wc.filename],
        x_cut = 512,
        y_cut = 512,
        z_cut = 25,
    envmodules:
        'matlab/R2019b'
    resources:
        partition = 'express',
        time="00:10:00",
        mem='32G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('iterative_biofilm_annotation/analysis')); """ + \
        """ tif2vtk('{output.vtk}', '{input.tif}', '{params.prediction}', {params.x_cut}, {params.y_cut}, {params.z_cut})" """
        