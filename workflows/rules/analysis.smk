
################
# Growth rates #
################

ruleorder: tracks2growthrateBiofilmQ > tracks2growthrate

#TODO(erjel): Is this rule still needed?
rule tracks2growthrateBiofilmQ:
    output:
        r"data\processed\tracks\{data}_model_BiofilmQ_growthrate.csv",
    input:
        tracks_csv = r"data\processed\tracks\{data}_model_BiofilmQ.csv",
        prediction_folder = r"data\interim\predictions\{data}\BiofilmQ",
        translations = r"data\interim\tracking\{data}_model_BiofilmQ_translations.csv",
        crop = r"data\interim\tracking\{data}_model_BiofilmQ_crop_offsets.csv",
    threads:
        1
    conda:
        "../envs/calc.yml"
    shell:
        r"python scripts\calc_growthrate.py {output} {input.tracks_csv} {input.prediction_folder} " +
        r"--transl_csv {input.translations} --crop_csv {input.crop}"
        

rule tracks2growthrate:
    output:
        r"data\processed\tracks\{data}_model_{model}_growthrate.csv",
    input:
        tracks_csv = r"data\processed\tracks\{data}_model_{model}.csv",
        prediction_folder = r"data\interim\predictions\{data}\{model}",
    wildcard_constraints:
        model="^(?!BiofilmQ$)"
    threads:
        1
    conda:
        "../envs/calc.yml"
    shell:
        r"python scripts\calc_growthrate.py {output} {input.tracks_csv} {input.prediction_folder}"

###############################
# Model prediction accuracies #
###############################

rule calc_accuracies_biofilmq:
    output:
        csv_file = "accuracies/data_{biofilmq_setting}/full_stacks_huy.csv",
    input:
        pred_path= "interim_data/predictions/full_stacks_huy/data_{biofilmq_setting}",
    params:
        # TODO(erjel): Where does this come from?
        gt_path = "data_BiofilmQ/full_stacks_huy/masks_intp",
    resources:
        partition = 'express',
        time="00:05:00",
        mem='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    conda:
        "../envs/stardist.yml"
    shell:
        "python iterative_biofilm_annotation/analysis/calc_accuracy_verbose.py" + \
        " {output} {input.pred_path} {params.gt_path}" + \
        " --pattern *Nz300.tif"

rule calc_accuracies:
    output:
        csv_file = "accuracies/{modelname}/{datasetname}.csv"
    wildcard_constraints:
        modelname = 'stardist_.*|horovod_.*|cellpose_.*'
    input:
        pred_path="interim_data/predictions/{datasetname}/test/images/{modelname}",
        gt_path="training_data/{datasetname}/test/masks"
    resources:
        partition = 'express',
        time="00:05:00",
        mem='16G',
        ntasks_per_node=1,
        ntasks_per_core=2,
        cpus_per_task=16,
    conda:
        "../envs/stardist.yml"
    shell:
        "python iterative_biofilm_annotation/analysis/calc_accuracy_verbose.py {output.csv_file} {input.pred_path} {input.gt_path}"