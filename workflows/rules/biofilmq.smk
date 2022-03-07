


# TODO(erjel): The accuracy for BiofilmQ can not be calculated directly on the prediction, but
# has to be calculated on an upsampled gt datasets ...
localrules: copy_prediction
rule copy_prediction:
    output:
        output_dir = directory('interim_data/predictions/full_stacks_huy/data_{biofilmq_settings}'),
    input:
        symlink = ".checkpoints/.symlink-data_BiofilmQ",
    params:
        input_tif = "data_BiofilmQ/full_stacks_huy/predictions/data_{biofilmq_settings}/Pos1_ch1_frame000001_Nz300.tif",
    wildcard_constraints:
        biofilmq_settings = "seeded_watershed|hartmann_et_al"
    shell:
        "mkdir -p {output.output_dir} && " + \
        "cp {params.input_tif} {output.output_dir}/"