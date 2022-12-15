
rule create_training_data:
    output:
        touch('training_data/.patches-semimanual-raw-{patch_size}.chkpt'),
    input:
        label_stack_dir = 'ZENODO/training_data/semi-manual',
        image_stack_dir = 'ZENODO/raw_data',
    params:
        output_dir = 'training_data/patches-semimanual-raw-{patch_size}',
        pattern  = '.*/biofilm_[2-5]_labels.tif', 
    conda:
        "../envs/poetry.yml"
    threads:
        1
    resources:
        time = '00:30:00', # TODO(erjel): Rough estimate
        mem_mb = 16000,
    shell:
        'python -m iterative_biofilm_annotation.data.create_dataset {params.output_dir}'
        ' {input.label_stack_dir} {input.image_stack_dir}'
        ' {params.pattern} {wildcards.patch_size}'


