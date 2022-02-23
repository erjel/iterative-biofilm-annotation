
ruleorder: stardist_merge_inference > stardist_inference # stardist_merge.smk vs stardist.smk

rule stardist_merge_inference:
    output:
        touch('interim_data/predictions/{data_folder}/{model_name}_merge/.chkpnt')
    input:
        folder="input_data/{data_folder}",
    # TODO(erjel): Make the model dependentcy explicit again
    params:
        model="models/{model_name}",
        output_dir="interim_data/predictions",
    threads:
        workflow.cores
    conda:
        r"../envs/stardist_merge.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/predict.py" + \
        " {input.folder}" + \
        " {params.model}" + \
        " {params.output_dir}/{wildcards.data_folder}" +\
        " --intp-factor 4" + \
        " --use-merge"