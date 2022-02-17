
rule train_stardist_model:
    output:
        directory("models/stardist_{n_rays}_{patchSize}_patches-{dataset_name}_{only_valid}_{percentage}prc_rep{replicate}")
    input:
        dataset="training_data/patches-{dataset_name}",
        output_symlink = "models/.symlink"
    wildcard_constraints:
        patchSize = '\d+x\d+x\d+'
    threads:
        workflow.cores
    resources:
        time="16:00:00",
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
        
        
rule stardist_prediction:
    output:
        touch(r'data\interim\predictions\{data_folder}\{model_name}\.chkpnt')
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
        r"python scripts\stardist_prediction.py {input.folder} {input.model} {params.output_dir}\{wildcards.data_folder} --intp-factor 4"
        
        
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