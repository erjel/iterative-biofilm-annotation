rule train_unet:
    output:
        directory("models/unet_{patchSize}_patches-{dataset_name}_rep{replicate}")
    input:
        dataset="training_data/patches-{dataset_name}",
    wildcard_constraints:
        patchSize = '\d+x\d+x\d+'
    threads:
        8
    resources:
        time="12:00:00",
        partition = config['slurm']['gpu_big'],
        constraint = "gpu",
        gres = config['slurm']['gpu_big_gres'],
        ntasks_per_core=2, # enable HT
        mem_mb='16G',
    conda:
        r"../envs/stardist.yml"
    shell:
        "python iterative_biofilm_annotation/unet/train.py" + \
        " {output}" + \
        " patches-{wildcards.dataset_name}" + \
        " {wildcards.patchSize}"
        
        
rule unet_testing:
    output:
        directory('interim_data/predictions/{data_folder}/{purpose}/{type}/{model_name}')
    input:
        folder="training_data/.{data_folder}.chkpt",
        model="models/{model_name}",
    wildcard_constraints:
        model_name = "unet_.*_rep\d+"
    params:
        folder="training_data/{data_folder}/{purpose}/{type}",
        output_dir="interim_data/predictions/{data_folder}/{purpose}/{type}",
    threads:
        8
    resources:
        partition = config['slurm']['gpu_big'],
        time = "00:15:00",
        constraint = "gpu",
        gres = config['slurm']['gpu_big_gres'],
        ntasks_per_core=2, # enable HT
        mem_mb='16G',
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python iterative_biofilm_annotation/unet/predict.py" + \
        " {params.output_dir}" + \
        " {input.model}" + \
        " {params.folder}"

rule unet_inference:
    output:
        touch('interim_data/predictions/{data_folder}/{model_name}/.chkpnt')
    wildcard_constraints:
        model_name = 'unet_.*_rep.*'
    input:
        model="models/{model_name}",
    params:
        folder = 'input_data/{data_folder}',
        output_dir="interim_data/predictions",
    conda:
        r"../envs/stardist.yml"
    threads:
        16
    resources:
        partition=config['slurm']['gpu_big'],
        time = "01:00:00",
        constraint = "gpu",
        gres = config['slurm']['gpu_big_gres'],
        ntasks_per_core=2, # enable HT
        mem='16G',
    shell:
        r"python iterative_biofilm_annotation/unet/predict.py" + \
        " {params.output_dir}/{wildcards.data_folder}" + \
        " {input.model}" + \
        " {params.folder}" + \
        " --input-pattern *_img.tif"