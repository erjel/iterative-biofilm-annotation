
ruleorder: stardist_merge_inference > stardist_inference # stardist_merge.smk vs stardist.smk

rule stardist_merge_inference:
    output:
        touch('interim_data/predictions/{data_folder}/{model_name}_merge/.chkpnt')
    input:
        symlink = "input_data/.symlink"
    # TODO(erjel): Make the model dependentcy explicit again
    params:
        model="models/{model_name}",
        output_dir="interim_data/predictions",
        folder="input_data/{data_folder}",
    threads:
        workflow.cores
    resources:
        partition = 'gpu_rtx5000',
        time = "24:00:00", # TODO(erjel): Max timelimit found reasonable one
        constraint = "gpu",
        gres = 'gpu:rtx5000:2',
        cpus_per_task=80,
        ntasks_per_core=2, # enable HT
        ntasks_per_node=1,
        mem='180G',
    conda:
        r"../envs/stardist_merge.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/predict.py" + \
        " {params.folder}" + \
        " {params.model}" + \
        " {params.output_dir}/{wildcards.data_folder}" +\
        " --intp-factor 4" + \
        " --use-merge"