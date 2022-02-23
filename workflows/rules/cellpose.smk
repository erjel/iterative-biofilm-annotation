
#ruleorder: horovod_cellpose_testing > stardist_testing

rule horovod_cellpose_testing:
    output:
        directory("interim_data/predictions/{datasetname}/horovod_cellpose_{modelname}")
    input:
        model_path="models/horovod_cellpose_{modelname}",
        dataset_path="training_data/{datasetname}",
    resources:
        partition="gpu1_rtx5000",
        gres="gpu:rtx5000:1",
        time="02:00:00"
    conda:
        "../envs/cellpose.yml"
    shell:
        "python scripts/cellpose_prediction.py {input.model_path} {input.dataset_path} {output}"