
#ruleorder: horovod_cellpose_testing > stardist_testing

rule cellpose_testing:
    output:
        directory("interim_data/predictions/{datasetname}/cellpose_{modelname}")
    input:
        dataset_path="training_data/{datasetname}",
        symlink = ".checkpoints/.symlink-models",
    params:
        # TODO(erjel): Make it an explizit dependency
        model_path="models/horovod_cellpose_{modelname}",
    resources:
        partition = config['slurm']['gpu_big'],
        constraint = 'gpu',
        gres = config['slurm']['gpu_big_gres'],
        time="01:00:00",
        mem='90G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=20,
        #time="02:00:00",
    conda:
        "../envs/cellpose.yml"
    shell:
        "python iterative_biofilm_annotation/cellpose/predict.py {params.model_path} {input.dataset_path} {output}"

localrules: download_models
rule download_models:
    output:
        "models.zip"
    shell:
        "wget https://datashare.mpcdf.mpg.de/s/95vEuRM9diRUJ3d/download -O {output}"

# TODO(erjel): Add the horovod training code instead of downloading zip files
"""
rule train_horovod_cellpose:
    output:
        directory("models/horovod_cellpose_{modelname}")
    input:
        dataset_chkpt="training_data/.{datasetname}.chkpt",
    params:
        model_path="models/horovod_cellpose_{modelname}",
    resources:
        partition = config['slurm']['gpu_big'],
        constraint = 'gpu',
        gres = config['slurm']['gpu_big_gres'],
        time="01:00:00",
        mem='90G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=20,
        #time="02:00:00",
    conda:
        "../envs/cellpose.yml"
    shell:
        "python iterative_biofilm_annotation/cellpose/train_horovod.py {input.dataset_path} {params.model_path} {output}"
"""
rule extract_horovod_models:
    output:
        directory("models/horovod_cellpose_{modelname}")
    input:
        "models.zip"
    params:
        zip_dir = "models/horovod_cellpose_{modelname}/*"
    shell:
        "unzip {input} {params.zip_dir} -d ./"    

rule horovod_cellpose_testing:
    output:
        directory("interim_data/predictions/{datasetname}/{purpose}/{type}/horovod_cellpose_{modelname}")
    input:
        model_path="models/horovod_cellpose_{modelname}",
        dataset_path="training_data/.{datasetname}.chkpt",
    params:
        dataset_path = "training_data/{datasetname}/{purpose}/{type}",
    resources:
        partition = config['slurm']['gpu_big'],
        constraint = 'gpu',
        gres = config['slurm']['gpu_big_gres'],
        time="01:00:00",
        mem_mb='96G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=20,
        #time="02:00:00",
    conda:
        "../envs/cellpose.yml"
    shell:
        "python iterative_biofilm_annotation/cellpose/predict.py {input.model_path} {params.dataset_path} {output}"