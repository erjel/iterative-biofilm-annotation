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

available_predictions = glob_wildcards('interim_data/predictions/full_semimanual-raw/test/images/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{replicate}_merge/im0.tif')
rule plot_fig3a:
    output:
        output_dir = directory('figures/fig3a'),
    input:
        cellpose_accuracies = expand(
            "accuracies/horovod_cellpose_patches-semimanual-raw-64x128x128_prc100_bs8_lr0.00625_wd0.00001_mt0.7_sge_rep{rep}/full_semimanual-raw.csv",
            rep = range(1, 6)
        ),
        # TODO(erjel): Optimal solution:
        stardist_accuracies = expand(
            "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}/full_semimanual-raw.csv",
            rep = [6, 7, 8, 10, 12]
        ),
        # Pragmatic solution:
        #stardist_accuracies = expand(
        #    "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}/full_semimanual-raw.csv",
        #    rep = available_predictions.replicate
        #),
        biofilmq_improved_accuracy = "accuracies/data_seeded_watershed/full_stacks_huy.csv",
        biofilmq_accuracy = "accuracies/data_hartmann_et_al/full_stacks_huy.csv",
        # TODO(erjel): Optimal solution:
        stardist_improved_accuracies = expand(
            "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}_merge/full_semimanual-raw.csv",
            rep = [6, 7, 8, 10, 12]
        ),
        # Pragmatic solution
        #stardist_improved_accuracies = expand(
        #    "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}_merge/full_semimanual-raw.csv",
        #    rep = available_predictions.replicate
        #),
        unet_accuracies = expand(
            "accuracies/unet_48x96x96_patches-semimanual-raw-64x128x128_rep{rep}/full_semimanual-raw.csv",
            rep = range(5)
        ),
        bcm3d_accuracies = expand(
            "accuracies/bcm3d_48x96x96_patches-semimanual-raw-64x128x128_v{rep}/full_semimanual-raw.csv",
            rep = range(5)
        ),
    params:
        labels = [
            'Stardist',
            'Cellpose',
            'Hartmann et al.',
            'Improved Hartmann et al.',
            'Stardist Improved',
            'Multi-class UNet',   
            'BCM3D 2.0',
        ]
    conda:
        "../envs/plot.yml"
    resources:
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
            " --biofilmq_accuracies {input.biofilmq_accuracy}"  + \
            " --stardist_improved_accuracies {input.stardist_improved_accuracies}" + \
            " --unet_accuracies {input.unet_accuracies}" + \
            " --bcm3d_accuracies {input.bcm3d_accuracies}" 


rule plot_fig3b:
    output:
        output_dir = directory('figures/fig3b'),
    input:
        cellpose_accuracies = expand(
            "accuracies/horovod_cellpose_patches-semimanual-raw-64x128x128_prc100_bs8_lr0.00625_wd0.00001_mt0.7_sge_rep{rep}/full_semimanual-raw.csv",
            rep = range(1, 6)
        ),
        # TODO(erjel): Optimal solution:
        stardist_accuracies = expand(
            "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}/full_semimanual-raw.csv",
            rep = [6, 7, 8, 10, 12]
        ),
        stardist_improved_accuracies = expand(
            "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}_merge/full_semimanual-raw.csv",
            rep = [6, 7, 8, 10, 12]
        ),
        # Pragmatic solution:
        #stardist_accuracies = expand(
        #    "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}/full_semimanual-raw.csv",
        #    rep = available_predictions.replicate
        #),
        #stardist_improved_accuracies = expand(
        #    "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}_merge/full_semimanual-raw.csv",
        #    rep = available_predictions.replicate
        #),
        biofilmq_improved_accuracies = "accuracies/data_seeded_watershed/full_stacks_huy.csv",
        biofilmq_accuracies = "accuracies/data_hartmann_et_al/full_stacks_huy.csv",
        unet_accuracies = expand(
            "accuracies/unet_48x96x96_patches-semimanual-raw-64x128x128_rep{rep}/full_semimanual-raw.csv",
            rep = range(5)
        ),
        bcm3d_accuracies = expand(
            "accuracies/bcm3d_48x96x96_patches-semimanual-raw-64x128x128_v{rep}/full_semimanual-raw.csv",
            rep = range(5)
        ),
    params:
        labels = [
            ' Cellpose',
            ' Stardist',
            ' Improved Hartmann et al.',
            ' Hartmann et al.',
            ' Stardist Improved',
            ' Multi-class UNet',
            ' BCM3D 2.0',
         ],
        plotstyle = [
            'dashed',
            'solid',
            'dashdot',
            'dotted',
            'solid',
            'dashed',
            'dashed',
        ]
    conda:
        "../envs/plot.yml",
    threads:
        16
    resources:
        time="00:05:00",
        mem_mb ='16G',
        ntasks_per_core=2,
    shell:
        "python iterative_biofilm_annotation/figures/fig3b_number_accuracies.py" + \
        " {output.output_dir}" + \
        " --labels {params.labels:q}" + \
        " --plotstyle {params.plotstyle}" + \
        " --cellpose_accuracies {input.cellpose_accuracies}" + \
        " --stardist_accuracies {input.stardist_accuracies}" + \
        " --biofilmq_improved_accuracies {input.biofilmq_improved_accuracies}" + \
        " --biofilmq_accuracies {input.biofilmq_accuracies}" + \
        " --stardist_improved_accuracies {input.stardist_improved_accuracies}" + \
        " --unet_accuracies {input.unet_accuracies}" + \
        " --bcm3d_accuracies {input.bcm3d_accuracies}" 
    

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

# Optimal solution:
stardist_models = expand("stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_{percentage}prc_rep{replicate}",
    percentage = [0.5, 0.8, 1, 5, 10, 25, 50, 70, 80, 90, 100],
    replicate = [6, 7, 8, 9, 10]
)
# Non working models:
non_working_models = [
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep9",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_1prc_rep10",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_25prc_rep9",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_50prc_rep10",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep10",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep7",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep6",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep7",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep8",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep9",
]

extra_models = [
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep12",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_1prc_rep15",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_25prc_rep11",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_50prc_rep13",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep13",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_5prc_rep16",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_70prc_rep12",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep13",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep14",
    "stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_80prc_rep15",
    ]

for m_del, m_add in zip(non_working_models, extra_models):
    stardist_models.remove(m_del)
    stardist_models.append(m_add)

# Pragmatic solution:
available_predictions = glob_wildcards('interim_data/predictions/full_semimanual-raw/test/images/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_{percentage}prc_rep{replicate}_merge/im0.tif')
#stardist_models = expand("stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_{percentage}prc_rep{replicate}",
#    zip,
#    percentage = available_predictions.percentage,
#    replicate = available_predictions.replicate,
#)

stardist_models_merge = [f'{stardist_model}_merge' for stardist_model in stardist_models]

def get_dependency_accs(model_list: List[str], dataset:str, wildcards, legacy: bool = False) -> List[str]:
    base_path = Path('accuracies_copy_bronto') if legacy else Path('accuracies')
    return [str(base_path / model / f'{dataset}.csv') for model in model_list]

get_dependency_accs_cellpose= partial(get_dependency_accs, cellpose_models_raw_full_low, 'accuracy_manual_raw_v3', legacy = True)
# TODO(erjel): Why not using:
# get_dependency_accs_cellpose = partial(get_dependency_accs, cellpose_models_raw_full_low, 'full_semimanual-raw')
get_dependency_accs_stardist = partial(get_dependency_accs, stardist_models, 'full_semimanual-raw')
get_dependency_accs_stardist_merge = partial(get_dependency_accs, stardist_models_merge, 'full_semimanual-raw')

rule plot_fig3c:
    output:
        output_dir = directory('figures/fig3c'),
    input:
        training_data_stardist = "training_data/patches-semimanual-raw-64x128x128",
        cellpose_accuracies = get_dependency_accs_cellpose,
        stardist_accuracies = get_dependency_accs_stardist,
        stardist_merge_accuracies = get_dependency_accs_stardist_merge,
    conda:
        "../envs/plot.yml",
    resources:
        time="00:05:00",
        mem_mb ='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    shell:
        "python iterative_biofilm_annotation/figures/fig3c_data_abundance_dependency.py" + \
        " {output.output_dir}"
        " {input.training_data_stardist}"
        " --cellpose_accuracies {input.cellpose_accuracies}"
        " --stardist_accuracies {input.stardist_accuracies}"
        " --stardist_merge_accuracies {input.stardist_merge_accuracies}"

rule create_fig3d_render:
    output:
        'figures/fig3d/{filename}_render.png'
    input:
        'interim_data/fn_fp_visualization/{filename}.vtk',
    params:
        'resources/paraview_render_save.pvsm',
    envmodules:
        'paraview/5.8',
    resources:
        partition = 'rvs',
        time="00:10:00",
        constraint = 'gpu',
        gres="gpu:rtx5000:1",
    shell:
        "vglrun pvpython iterative_biofilm_annotation/figures/fig3d_render_vtk.py {input} {output} {params}"
        

rule plot_figS4:
    output:
        output_dir = directory('figures/figS4'),
    input:
        cellpose_accuracies = expand(
            "accuracies/horovod_cellpose_patches-semimanual-raw-64x128x128_prc100_bs8_lr0.00625_wd0.00001_mt0.7_sge_rep{rep}/manual_raw_v3.csv",
            rep = range(1, 6)
        ),
        stardist_accuracies = expand(
            "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}/manual_raw_v3.csv",
            rep = [6, 7, 8, 10, 12]
        ),
        biofilmq_improved_accuracy = "accuracies/data_seeded_watershed/manual_raw_v3.csv",
        biofilmq_accuracy = "accuracies/data_hartmann_et_al/manual_raw_v3.csv",
        stardist_improved_accuracies = expand(
            "accuracies/stardist_192_48x96x96_patches-semimanual-raw-64x128x128_True_100prc_rep{rep}_merge/manual_raw_v3.csv",
            rep = [6, 7, 8, 10, 12]
        ),
        semimanual_accuracy = "accuracies/semi_manual/manual_raw_v3.csv",
    params:
        labels = [
            'Stardist',
            'Cellpose',
            'Hartmann et al.',
            'Improved Hartmann et al.',
            'Stardist improved',
            'Semi-manual annotation',   
        ]
    conda:
        "../envs/plot.yml"
    resources:
        time="00:05:00",
        mem_mb='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    shell:
        "python iterative_biofilm_annotation/figures/figS4_results_vs_manual_annotation.py" + \
            " {output.output_dir}" + \
            " --labels {params.labels:q}" + \
            " --stardist_accuracies {input.stardist_accuracies}" + \
            " --cellpose_accuracies {input.cellpose_accuracies}" + \
            " --biofilmq_accuracies {input.biofilmq_accuracy}"  + \
            " --biofilmq_improved_accuracies {input.biofilmq_improved_accuracy}" + \
            " --stardist_improved_accuracies {input.stardist_improved_accuracies}" + \
            " --semimanual_accuracies {input.semimanual_accuracy}"
