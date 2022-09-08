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
        time="24:00:00",
        partition = 'gpu1_rtx5000',
        constraint = "gpu",
        gres = 'gpu:rtx5000:1',
        ntasks_per_core=2, # enable HT
        mem_mb='64G',
    conda:
        r"../envs/stardist.yml"
    shell:
        "python iterative_biofilm_annotation/unet/train.py" + \
        " {output}" + \
        " patches-{wildcards.dataset_name}" + \
        " {wildcards.patchSize}"
        
        
rule unet_testing:
    output:
        directory('interim_data/predictions/{data_folder}/{model_name}')
    input:
        folder="training_data/{data_folder}",
        model="models/{model_name}",
    wildcard_constraints:
        model_name = "unet_.*_rep\d+"
    params:
        output_dir="interim_data/predictions",
    threads:
        8
    resources:
        partition='gpu1_rtx5000',
        time = "00:15:00",
        constraint = "gpu",
        gres = 'gpu:rtx5000:1',
        ntasks_per_core=2, # enable HT
        mem_mb='16G',
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python iterative_biofilm_annotation/unet/predict.py" + \
        " {params.output_dir}/{wildcards.data_folder}" + \
        " {input.model}" + \
        " {input.folder}"

rule unet_inference:
    output:
        touch('interim_data/predictions/{data_folder}/{model_name}/.chkpnt')
    input:
        model="models/{model_name}",
        folder="input_data/{data_folder}",
    params:
        output_dir="interim_data/predictions",
    conda:
        r"../envs/stardist.yml"
    threads:
        16
    resources:
        partition='gpu1_rtx5000',
        time = "01:00:00",
        constraint = "gpu",
        gres = 'gpu:rtx5000:1',
        ntasks_per_core=2, # enable HT
        mem='64G',
    shell:
        r"python iterative_biofilm_annotation/unet/predict.py" + \
        " {params.output_dir}/{wildcards.data_folder}" + \
        " {input.model}" + \
        " {input.folder}"