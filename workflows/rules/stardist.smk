
rule train_stardist_model:
    output:
        directory("models/stardist_192_{patch_size}_{datasetname}_True_100prc_rep{rep_number}")
    input:
        dataset_dir = "training_data/{datasetname}",
        output_symlink = "models/.symlink"
    wildcard_constraints:
        patch_size = '\d+x\d+x\d+'
    resources:
        nvidia_gpu=1
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python iterative_biofilm_annotation/stardist/train.py {output} {input.dataset_dir}" + \
        " --patch_size {wildcards.patch_size}"
        
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