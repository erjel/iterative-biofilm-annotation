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
        directory("models/bcmd3d_{patchSize}_{dataset_name}_{target}_v{version}")
    wildcard_constraints:
        patchSize = '\d+x\d+x\d+',
        target = '(1|2)',
    input:
        "training_data/{dataset_name}/train/target_bcm3d_{target}",
        "training_data/{dataset_name}/valid/target_bcm3d_{target}",
    params:
        modeldir = 'models',
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
        " {params.modeldir}" +\
        " {wildcards.dataset_name}" +\
        " {params.target}" +\
        " {wildcards.patchSize}"    

rule bcm3d_post_processing:
    output:
        touch('interim_data/predictions/{data_folder}/{model_name}/.chkpnt')
    wildcard_constraints:
        model_name = 'bcm3d_'
    input:
        edt = 'interim_data/bcm3d_cnn_output/{data_folder}/{model_name}/.edt_chkpt',
        cell_bdy = 'interim_data/bcm3d_cnn_output/{data_folder}/{model_name}/.bdy_chkpt',
    params:
        folder = 'training_data/{data_folder}',
        output_dir="interim_data/predictions",
    conda:
        r"../envs/bcm3d.yml"
    threads:
        16
    resources:
        time = "01:00:00",
        ntasks_per_core=2, # enable HT
        mem='16G',
    shell:
        r"python iterative_biofilm_annotation/bcm3d/post_processing.py" + \
        " {params.output_dir}/{wildcards.data_folder}" + \
        " {input.model}" + \
        " {params.folder}" + \
        " --input-pattern *_img.tif"
