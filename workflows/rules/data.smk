
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

rule create_test_data_full:
    output:
        touch('training_data/.full_semimanual-raw.chkpt'),
    input:
        label_stack_dir = 'ZENODO/training_data/semi-manual',
        image_stack_dir = 'ZENODO/raw_data',
    params:
        output_dir = 'training_data/full_semimanual-raw',
        pattern  = '.*/biofilm_1_labels.tif',
        patch_size = 'all',
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
        ' {params.pattern} {params.patch_size} --test-only'

use rule create_test_data_full as create_test_data_manual with:
    output:
        touch('training_data/.manual_raw_v3.chkpt'),
    input:
        label_stack_dir = 'ZENODO/training_data/manual',
        image_stack_dir = 'ZENODO/training_data/manual',
    params:
        output_dir = 'training_data/manual_raw_v3',
        pattern  = '.*/biofilm_1_cropped_labels.tif',
        patch_size = 'all',