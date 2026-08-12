
rule train_stardist_model:
    output:
        directory("models/stardist_{n_rays}_{patchSize}_patches-{dataset_name}_{only_valid}_{percentage}prc_rep{replicate}")
    input:
        dataset="training_data/.patches-{dataset_name}.chkpt",
    wildcard_constraints:
        patchSize = '\d+x\d+x\d+'
    threads:
        40
    resources:
        # Single GPU OOM for 
        time="24:00:00",
        partition = config['slurm']['gpu_big'],
        constraint = "gpu",
        gres = config['slurm']['gpu_big_gres'],
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
        directory('interim_data/predictions/{data_folder}/{purpose}/{type}/{model_name}')
    wildcard_constraints:
        model_name = "stardist_.*_rep\d+"
    input:
        model = "models/{model_name}",
        input_chkpt = "training_data/.{data_folder}.chkpt",
    params:
        input_dir = "training_data/{data_folder}/{purpose}/{type}",
    threads:
        40
    resources:
        partition = config['slurm']['gpu_big'],
        time = "1:00:00", # Measured 20 min
        constraint = "gpu",
        gres = config['slurm']['gpu_big_gres'],
        cpus_per_task=80,
        ntasks_per_core=2, # enable HT
        ntasks_per_node=1,
        mem_mb='150G', # Measured 115G
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/predict.py" + \
        " {params.input_dir}" + \
        " {input.model}" + \
        " {output}"

ruleorder: stardist_merge_inference > stardist_inference # stardist_merge.smk vs stardist.smk

rule stardist_inference:
    output:
        touch('interim_data/predictions/{data_folder}/{model_name}/.chkpnt')
    wildcard_constraints:
        model_name = 'stardist_*'
    input:
        input_chkpt = "input_data/.{data_folder}.chkpt",
        model="models/{model_name}",
    params:
        output_dir="interim_data/predictions/{data_folder}/{model_name}",
        input_dir="input_data/{data_folder}",
    threads:
        40
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/predict.py" + \
        " {params.input_dir}" + \
        " {params.model}" + \
        " {params.output_dir}" +\
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
