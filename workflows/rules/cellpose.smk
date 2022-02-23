
#ruleorder: horovod_cellpose_testing > stardist_testing

rule horovod_cellpose_testing:
    output:
        directory("interim_data/predictions/{datasetname}/horovod_cellpose_{modelname}")
    input:
        dataset_path="training_data/{datasetname}",
    params:
        # TODO(erjel): Make it an explizit dependency
        model_path="models/horovod_cellpose_{modelname}",
    resources:
        constraint = 'gpu',
        gres="gpu:rtx5000:1",
        time="01:00:00",
        mem='90G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=20,
        #time="02:00:00",
    conda:
        "../envs/cellpose.yml"
    shell:
        "python iterative_biofilm_annotation/cellpose/predict.py {params.model_path} {input.dataset_path} {output}"