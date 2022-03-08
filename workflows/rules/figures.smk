from functools import partial
from typing import List
from pathlib import Path


rule create_tiffs_with_track_id:
    output:
        directory('data/processed/tracks/{data}_model_{model}_tif')
    input:
        tracks_csv = r"data\processed\tracks\{data}_model_{model}.csv",
        prediction_folder = r"data\interim\predictions\{data}\{model}",
    threads:
        1
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tracks2tif('{output}', '{input.tracks_csv}', '{input.prediction_folder}')" """
        

rule create_vtks_with_track_id:
    output:
        directory('data/processed/tracks/{data}_model_{model}_vtk')
    input:
        tracks_csv = r"data\processed\tracks\{data}_model_{model}.csv",
        prediction_folder = r"data\interim\predictions\{data}\{model}",
    threads:
        1
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tracks2vtk('{output}', '{input.tracks_csv}', '{input.prediction_folder}')" """

rule plot_growthrate_heatmap:
    output:
        r'reports\figures\{data}\{modelname}_growthrate_heatmap.png',
    input:
        r'data\processed\tracks\{data}_model_{modelname}_growthrate.csv',
    conda:
        r'envs\plot.yml'
    shell:
        r"python scripts\plot_growthrate_heatmap.py {output} {input}"

rule plot_growthrate:
    output:
        r'reports\figures\{data}\{modelname}_single_cell_growthrate.png',
    input:
        r'data\processed\tracks\{data}_model_{modelname}_growthrate.csv',
    conda:
        r'envs\calc.yml'
    shell:
        r"python scripts\plot_growthrate.py {output} {input}"

rule plot_fig3a:
    output:
        output_dir = directory('figures/fig3a'),
    input:
        cellpose_accuracies = expand(
            "accuracies/horovod_cellpose_patches-semimanual-raw-64x128x128_prc100_bs8_lr0.00625_wd0.00001_mt0.7_sge_rep{rep}/full_semimanual-raw.csv",
            rep = range(1, 6)
        ),
        stardist_accuracies = expand(
            "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}/full_semimanual-raw.csv",
            rep = range(1, 6)
        ),
        biofilmq_improved_accuracy = "accuracies/data_seeded_watershed/full_stacks_huy.csv",
        biofilmq_accuracy = "accuracies/data_hartmann_et_al/full_stacks_huy.csv",
    params:
        labels = [
            'Improved Hartmann et al.',
            'Hartmann et al.',
            'Cellpose',
            'Stardist',
        ]
    conda:
        "../envs/plot.yml"
    resources:
        partition = 'express',
        time="00:05:00",
        mem='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    shell:
        "python iterative_biofilm_annotation/figures/fig3a_segmentation_comparision.py" + \
            " {output.output_dir}" + \
            " --labels {params.labels:q}" + \
            " --stardist_accuracies {input.stardist_accuracies}" + \
            " --cellpose_accuracies {input.cellpose_accuracies}" + \
            " --biofilmq_improved_accuracies {input.biofilmq_improved_accuracy}" + \
            " --biofilmq_accuracies {input.biofilmq_accuracy}" # + \
            # " --stardist_improved_accuracies {input.stardist_improved_accuracies}"


rule plot_fig3b:
    output:
        output_dir = directory('figures/fig3b'),
    input:
        # TODO(erjel): Use Range instead of single value
        #cellpose_accuracies = expand(
        #    "accuracies/horovod_cellpose_patches-semimanual-raw-64x128x128_prc100_bs8_lr0.00625_wd0.00001_mt0.7_sge_rep{rep}/full_semimanual-raw.csv",
        #    rep = range(1, 6)
        #),
        #stardist_accuracies = expand(
        #    "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}/full_semimanual-raw.csv",
        #    rep = range(1, 6)
        #),
        cellpose_accuracies = "accuracies/horovod_cellpose_patches-semimanual-raw-64x128x128_prc100_bs8_lr0.00625_wd0.00001_mt0.7_sge_rep1/full_semimanual-raw.csv",
        stardist_accuracies = "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep5/full_semimanual-raw.csv",
        biofilmq_improved_accuracies = "accuracies/data_seeded_watershed/full_stacks_huy.csv",
        biofilmq_accuracies = "accuracies/data_hartmann_et_al/full_stacks_huy.csv",
    params:
        labels = [' Stardist', ' Improved Hartmann et al.', ' Hartmann et al.', ' Cellpose'],
        plotstyle = ['solid', 'dashed', 'dashdot', 'dotted']
    conda:
        "../envs/plot.yml",
    resources:
        partition = 'express',
        time="00:05:00",
        mem ='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    shell:
        "python iterative_biofilm_annotation/figures/fig3b_number_accuracies.py" + \
        " {output.output_dir}" + \
        " --labels {params.labels:q}" + \
        " --plotstyle {params.plotstyle}" + \
        " --cellpose_accuracies {input.cellpose_accuracies}" + \
        " --stardist_accuracies {input.stardist_accuracies}" + \
        " --biofilmq_improved_accuracies {input.biofilmq_improved_accuracies}" + \
        " --biofilmq_accuracies {input.biofilmq_accuracies}" 
    

cellpose_models_raw_full_low = [
    "cellpose_patches-semimanual-raw-64x128x128_True_100prc_rep1_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_100prc_rep2_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_100prc_rep3_ep500_dep50",
    #"cellpose_patches-semimanual-raw-64x128x128_True_100prc_rep4_ep500_dep50",
    #"cellpose_patches-semimanual-raw-64x128x128_True_100prc_rep5_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_90prc_rep1_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_90prc_rep2_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_90prc_rep3_ep500_dep50",
    #"cellpose_patches-semimanual-raw-64x128x128_True_90prc_rep4_ep500_dep50",
    #"cellpose_patches-semimanual-raw-64x128x128_True_90prc_rep5_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_80prc_rep1_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_80prc_rep2_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_80prc_rep3_ep500_dep50",
    #"cellpose_patches-semimanual-raw-64x128x128_True_80prc_rep4_ep500_dep50",
    #"cellpose_patches-semimanual-raw-64x128x128_True_80prc_rep5_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_70prc_rep1_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_70prc_rep2_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_70prc_rep3_ep500_dep50",
    #"cellpose_patches-semimanual-raw-64x128x128_True_70prc_rep4_ep500_dep50",
    #"cellpose_patches-semimanual-raw-64x128x128_True_70prc_rep5_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_50prc_rep1_ep500_dep50",
    "cellpose_patches-semimanual-raw-64x128x128_True_50prc_rep2_ep500_dep100",
    "cellpose_patches-semimanual-raw-64x128x128_True_50prc_rep3_ep500_dep100",
    #"cellpose_patches-semimanual-raw-64x128x128_True_50prc_rep4_ep500_dep100",
    #"cellpose_patches-semimanual-raw-64x128x128_True_50prc_rep5_ep500_dep100",
    "cellpose_patches-semimanual-raw-64x128x128_True_25prc_rep1_ep500_dep125",
    "cellpose_patches-semimanual-raw-64x128x128_True_25prc_rep2_ep500_dep125",
    "cellpose_patches-semimanual-raw-64x128x128_True_25prc_rep3_ep500_dep125",
    #"cellpose_patches-semimanual-raw-64x128x128_True_25prc_rep4_ep500_dep500",
    #"cellpose_patches-semimanual-raw-64x128x128_True_25prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_10prc_rep1_ep500_dep250",
    "cellpose_patches-semimanual-raw-64x128x128_True_10prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_10prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_10prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_10prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_9prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_9prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_9prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_9prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_9prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_8prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_8prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_8prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_8prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_8prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_7prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_7prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_7prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_7prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_7prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_6prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_6prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_6prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_6prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_6prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_5prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_5prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_5prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_5prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_5prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_4prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_4prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_4prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_4prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_4prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_3prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_3prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_3prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_3prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_3prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_2prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_2prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_2prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_2prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_2prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_1prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_1prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_1prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_1prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_1prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.8prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.8prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.8prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.8prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.8prc_rep5_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.5prc_rep1_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.5prc_rep2_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.5prc_rep3_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.5prc_rep4_ep500_dep500",
    "cellpose_patches-semimanual-raw-64x128x128_True_0.5prc_rep5_ep500_dep500",
]

stardist_models = [
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_90prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_90prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_90prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_90prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_90prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_50prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_50prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_50prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_50prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_50prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_25prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_25prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_25prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_25prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_25prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_10prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_10prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_10prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_10prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_10prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_1prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_1prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_1prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_1prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_1prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.8prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.8prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.8prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.8prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.8prc_rep5",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.5prc_rep1",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.5prc_rep2",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.5prc_rep3",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.5prc_rep4",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_0.5prc_rep5",
]

def get_dependency_accs(model_list: List[str], dataset:str, wildcards) -> List[str]:
    return [str(Path('accuracies_copy_bronto') / model / f'{dataset}.csv') for model in model_list]
    # TODO(erjel): Use just calculated accuracies instead of copies from bronto
    #return [str(Path('accuracies') / model / f'{dataset}.csv') for model in model_list]

get_dependency_accs_cellpose = partial(get_dependency_accs, cellpose_models_raw_full_low, 'accuracy_manual_raw_v3')
# TODO(erjel): Why not using:
# get_dependency_accs_cellpose = partial(get_dependency_accs, cellpose_models_raw_full_low, 'full_semimanual-raw')
get_dependency_accs_stardist = partial(get_dependency_accs, stardist_models, 'accuracy_full_semimanual-raw')

rule plot_fig3c:
    output:
        output_dir = directory('figures/fig3c'),
    input:
        training_data_stardist = "training_data/patches-semimanual-raw-64x128x128",
        cellpose_accuracies = get_dependency_accs_cellpose,
        stardist_accuracies = get_dependency_accs_stardist,
    conda:
        "../envs/plot.yml",
    resources:
        partition = 'express',
        time="00:05:00",
        mem ='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    shell:
        "python iterative_biofilm_annotation/figures/fig3c_data_abundance_dependency.py" + \
        " {output.output_dir}"
        " {input.training_data_stardist}"
        " --cellpose_accuracies {input.cellpose_accuracies}"
        " --stardist_accuracies {input.stardist_accuracies}"
