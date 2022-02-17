
ruleorder: tracks2growthrateBiofilmQ > tracks2growthrate

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