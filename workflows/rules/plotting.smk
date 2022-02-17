
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