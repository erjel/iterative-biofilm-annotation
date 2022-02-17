
rule biofilmQ2trackmate:
    output:
        xml = r"data\interim\tracking\{data}_model_BiofilmQ.xml",
        trans = r"data\interim\tracking\{data}_model_BiofilmQ_translations.csv",
        crop = r"data\interim\tracking\{data}_model_BiofilmQ_crop_offsets.csv",
    input:
        data_folder = r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\data',
        int_data_path = r'data\interim\tracking\{data}.tif',
    threads:
        1
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); biofilmQ2trackMate('{output.xml}',  '{output.trans}', '{output.crop}', '{input.int_data_path}', '{input.data_folder}')" """
    

rule labelimages2trackmate:
    output:
        "interim_data/tracking/{data}_model_{model}.xml",
    input:
        int_data_path = 'interim_data/trackmate_stacks/{data}.tif',
        input_folder = 'interim_data/predictions/{data}/{model}',
    conda:
        "../envs/jinja2.yml"
    resources:
        partition='',
        constraint='',
        gres='',
        ntasks=1,
        cpu_per_task=1,
        ntasks_per_node=1,
        mem=32000,
        time='02:00:00',
    shell:
        "python iterative_biofilm_annotation/trackmate/labelimage2trackmate.py" + \
        " --int_data_path {input.int_data_path}" + \
        " --input_folder {input.input_folder}" + \
        " --output_xml {output}"
        
rule stack4trackmate:
    output:
        "interim_data/trackmate_stacks/{data}.tif"
    input:
        "input_data/{data}"
    conda:
        "../envs/stardist.yml"
    resources:
        partition='',
        constraint='',
        gres='',
        ntasks=1,
        cpu_per_task=1,
        ntasks_per_node=1,
        mem=32000,
        time="00:15:00",
    shell:
        "python iterative_biofilm_annotation/trackmate/create_stack_for_trackmate.py {output} {input}"
                
          
rule trackmate2napari:
    output:
        "tracks/{data}_model_{model}.csv"
    input:
        "interim_data/tracking/{data}_model_{model}_Tracks.xml" # comes from manual TrackMate step
    conda:
        "../envs/stardist.yml"
    resources:
        partition='',
        constraint='',
        gres='',
        ntasks=1,
        cpu_per_task=1,
        ntasks_per_node=1,
        mem=32000,
        time="00:15:00",
    shell:
        "python iterative_biofilm_annotation/trackmate_xml_to_napari_csv.py {input} {output}"