rule bcm3d_data_preparation:
    output:
        directory('training_data/{dataset_name}/{purpose}/target_bcm3d_1'),
        directory('training_data/{dataset_name}/{purpose}/target_bcm3d_2'),
    input:
        dataset_dir = 'training_data/.{dataset_name}.chkpt',
    params:
        mask_dir = 'training_data/{dataset_name}/{purpose}/masks'
    conda:
        "../envs/bcm3d_prep.yml"
    threads:
        16
    resources:
        time = "00:30:00",
        mem_mb = "16000",
    shell:
        "python -u iterative_biofilm_annotation/bcm3d/data_preparation.py" + \
        " {params.mask_dir}"

rule bcm3d_training:
    output:
        directory("models/bcm3d_{patchSize}_{dataset_name}_{target}_v{version}")
    wildcard_constraints:
        patchSize = '\d+x\d+x\d+',
        target = '(1|2)',
    input:
        "training_data/{dataset_name}/train/target_bcm3d_{target}",
        "training_data/{dataset_name}/valid/target_bcm3d_{target}",
    params:
        target = 'target_bcm3d_{target}',
    threads:
        4
    resources:
        time = "12:00:00",
        partition = gpu_big,
        constraint = 'gpu',
        gres = gpu_big_gres,
        ntasks_per_core = 2,
        mem_mb = '16000',
    conda:
        '../envs/stardist_new.yml',
    shell:
        "python -u iterative_biofilm_annotation/bcm3d/train.py" + \
        " {output}" +\
        " {wildcards.dataset_name}" +\
        " {params.target}" +\
        " {wildcards.patchSize}"    

rule bcm3d_predict:
    output:
        directory('interim_data/bcm3d_cnn_output/{data_folder}/{purpose}/{type}/{model_name}'),
    wildcard_constraints:
        model_name = 'bcm3d_.*'
    input:
        model = 'models/{model_name}',
        dataset_dir = 'training_data/.{data_folder}.chkpt',
    params:
        folder = 'training_data/{data_folder}/{purpose}/{type}',
        output_dir="interim_data/bcm3d_cnn_output",
    conda:
        r"../envs/stardist_new.yml"
    shell:
        "python -u iterative_biofilm_annotation/bcm3d/predict.py"
        " {params.output_dir}/{wildcards.data_folder}"
        " {input.model}"
        " {params.folder}"


rule bcm3d_post_processing:
    output:
        directory('interim_data/predictions/{data_folder}/bcm3d_{patch_size}_{dataset_name}_v{version}'),
    wildcard_constraints:
        model_name = 'bcm3d_.*'
    input:
        edt_dir = 'interim_data/bcm3d_cnn_output/{data_folder}/bcm3d_{patch_size}_{dataset_name}_1_v{version}',
        cell_bdy_dir = 'interim_data/bcm3d_cnn_output/{data_folder}/bcm3d_{patch_size}_{dataset_name}_2_v{version}',
    conda:
        r"../envs/bcm3d_prep.yml"
    threads:
        16
    resources:
        time = "01:00:00",
        ntasks_per_core=2, # enable HT
        mem='16G',
    shell:
        r"python iterative_biofilm_annotation/bcm3d/post_processing.py"
        " {output}"
        " {input.edt_dir}"
        " {input.cell_bdy_dir}"
        " --input-pattern *_img.tif"
