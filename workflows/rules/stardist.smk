
rule train_stardist_model:
    output:
        directory("models/stardist_{n_rays}_{patchSize}_patches-{dataset_name}_{only_valid}_{percentage}prc_rep{replicate}")
    input:
        dataset="training_data/patches-{dataset_name}",
        #output_symlink = "models/.symlink"
    wildcard_constraints:
        patchSize = '\d+x\d+x\d+'
    threads:
        40
    resources:
        # Single GPU OOM for 
        time="24:00:00",
        partition = 'gpu1_rtx5000',
        constraint = "gpu",
        gres = 'gpu:rtx5000:1',
        cpus_per_task=40,
        ntasks_per_core=2, # enable HT
        ntasks_per_node=1,
        mem_mb='90G',
    conda:
        r"../envs/stardist.yml"
    shell:
        "python iterative_biofilm_annotation/stardist/train.py" + \
        " stardist_{wildcards.n_rays}_{wildcards.patchSize}_patches-{wildcards.dataset_name}_{wildcards.only_valid}_{wildcards.percentage}prc_rep{wildcards.replicate}" + \
        " models" + \
        " patches-{wildcards.dataset_name}" + \
        " {wildcards.n_rays}" + \ 
        " {wildcards.patchSize}" + \
        " {wildcards.only_valid}" + \
        " --percentage {wildcards.percentage}"
        
        
rule stardist_testing:
    output:
        directory('interim_data/predictions/{data_folder}/{model_name}')
    input:
        folder="training_data/{data_folder}",
        model="models/{model_name}",
    wildcard_constraints:
        model_name = "stardist_.*_rep\d+"
    params:
        output_dir="interim_data/predictions",
    threads:
        40
    resources:
        partition='gpu_rtx5000',
        time = "1:00:00", # Measured 20 min
        constraint = "gpu",
        gres = 'gpu:rtx5000:2',
        cpus_per_task=80,
        ntasks_per_core=2, # enable HT
        ntasks_per_node=1,
        mem_mb='150G', # Measured 115G
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/predict.py" + \
        " {input.folder}" + \
        " {input.model}" + \
        " {params.output_dir}/{wildcards.data_folder}"

ruleorder: stardist_merge_inference > stardist_inference # stardist_merge.smk vs stardist.smk

rule stardist_inference:
    output:
        touch('interim_data/predictions/{data_folder}/{model_name}/.chkpnt')
    wildcard_constraints:
        model_name = 'stardist_*'
    input:
        symlink = "input_data/.symlink",
    # TODO(erjel): Make the model dependentcy explicit again
    params:
        model="models/{model_name}",
        output_dir="interim_data/predictions",
        folder="input_data/{data_folder}",
    threads:
        workflow.cores
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/predict.py" + \
        " {params.folder}" + \
        " {params.model}" + \
        " {params.output_dir}\{wildcards.data_folder}" +\
        " --intp-factor 4"
        
# TODO(erjel): Is this rule required at some point?        
rule stardist_prediction_probabilities:
    output:
        directory(r'data\interim\predictions\{data_folder}\{model_name}\probs')
    input:
        folder=r"Y:\Eric\prediction_test\data\interim\{data_folder}",
        model=r"models\{model_name}"
    params:
        output_dir=r"data\interim\predictions"
    threads:
        workflow.cores
    resources:
        nvidia_gpu=1
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python scripts\stardist_prediction.py {input.folder} {input.model} {params.output_dir}\{wildcards.data_folder} --intp-factor 4 --probs"
