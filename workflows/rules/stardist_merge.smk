#TODO(erjel): Save UNet prediction on disk before you use the merge code in order utilize the GPU nodes more efficiently

slurm_config = config.get("slurm")

if slurm_config:
    gpu_big = slurm_config.get("gpu_big", "gpu_rtx5000")
    gpu_big_gres = slurm_config.get("gpu_big_gres", "gpu:rtx5000:2")
    gpu_time_limit = slurm_config.get("gpu_time_limit", "rule-dependent")
else:
    gpu_big = "gpu_rtx5000"
    gpu_big_gres = "gpu:rtx5000:2"
    gpu_time_limit = "rule-dependent"

ruleorder:
    stardist_merge_inference > stardist_testing > stardist_inference # stardist_merge.smk vs stardist.smk

rule stardist_merge_inference:
    output:
        touch('interim_data/predictions/{data_folder}/{model_name}_merge/.chkpnt')
    input:
        folder="input_data/{data_folder}",
    # TODO(erjel): Make the model dependency explicit again
    params:
        model="models/{model_name}",
        output_dir="interim_data/predictions",
    threads:
        40
    resources:
        partition = 'gpu_rtx5000',
        time = "24:00:00", # TODO(erjel): Max timelimit; get reasonable one
        constraint = "gpu",
        gres = 'gpu:rtx5000:2',
        cpus_per_task=80,
        ntasks_per_core=2, # enable HT
        ntasks_per_node=1,
        mem_mb='180G',
    conda:
        r"../envs/stardist_merge.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/predict.py" + \
        " {input.folder}" + \
        " {params.model}" + \
        " {params.output_dir}/{wildcards.data_folder}" +\
        " --intp-factor 4" + \
        " --use-merge"

# TODO(erjel): Unify stardist_merge_testing and stardist_merge_inference. Possible ideas:
# - save predictions for input_data in a different folder (i.e. "inference") instead of "prediction"
ruleorder:
    stardist_merge_testing > stardist_merge_inference > stardist_testing > stardist_inference

rule stardist_merge_testing:
    output:
        directory('interim_data/predictions/{data_folder}/{model_name}_merge')
    input:
        #symlink = ".checkpoints/.symlink-training_data"
        #model="models/{model_name}",
    params:
        model="models/{model_name}",
        output_dir= lambda wc: "interim_data/predictions",
        folder="training_data/{data_folder}",
    threads:
        40
    resources:
        partition = gpu_big,
        time = "03:00:00" if gpu_time_limit == 'rule-dependent' else gpu_time_limit,
        gres = gpu_big_gres,
        constraint = "gpu",
        cpus_per_task=80,
        ntasks_per_core=2, # enable HT
        ntasks_per_node=1,
        mem_mb='180G',
    conda:
        r"../envs/stardist_merge.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/predict.py" + \
        " {params.folder}" + \
        " {params.model}" + \
        " {params.output_dir}/{wildcards.data_folder}" +\
        " --use-merge"
