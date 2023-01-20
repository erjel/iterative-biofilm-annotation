localrules: download_biofilmq_data # BiofilmQ single-cell is not published
rule download_biofilmq_data:
    output:
        "data_BiofilmQ.zip"
    shell:
        "wget https://datashare.mpcdf.mpg.de/s/MIVweVPvg50Q388/download -O {output}"




# TODO(erjel): The accuracy for BiofilmQ can not be calculated directly on the prediction, but
# has to be calculated on an upsampled gt datasets ...
localrules: copy_prediction
rule copy_prediction:
    output:
        output_dir = directory('interim_data/predictions/full_stacks_huy/data_{biofilmq_settings}'),
    input:
        "data_BiofilmQ.zip",
    params:
        zip_path = "data_BiofilmQ/full_stacks_huy/predictions/data_{biofilmq_settings}",
    wildcard_constraints:
        biofilmq_settings = "seeded_watershed|hartmann_et_al"
    shell:
        "unzip -p {input} {params.zip_path} > {output}"

localrules: create_biofilmq_manual_raw_v3_prediction
rule create_biofilmq_manual_raw_v3_prediction:
    output:
        directory('interim_data/predictions/manual_raw_v3/test/images/data_{biofilmq_settings}'),
    input:
        "interim_data/predictions/full_stacks_huy/data_{biofilmq_settings}-downsampled/Pos1_ch1_frame000001_Nz300.tif",
    conda: 
        "../envs/calc.yml"
    shell:
        "python iterative_biofilm_annotation/analysis/crop_biofilmq.py {output} {input}"