
rule train_stardist_model:
    output:
        directory("models/stardist_192_48x96x96_{datasetname}_rep{rep_nummer}")
    input:
        "training_data/{datasetname}",
        "models/.symlink"
    resources:
        nvidia_gpu=1
    conda:
        r"../envs/stardist.yml"
    shell:
        r"python scripts/stardist_training.py {output} {input}"
        
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