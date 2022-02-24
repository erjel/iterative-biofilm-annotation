
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
        'dummy',
    conda:
        "../envs/plot.yml",
    shell:
        "python iterative_biofilm_annotation/figures/figb_number_accuracies.py" 
    


