#set CUDA_VISIBLE_DEVICES=1 && snakemake -j --use-conda --resources nvidia_gpu=1

include: r"workflows\snakefile_care"

from pathlib import Path

rule all:
	input:
		expand(r"data\interim\vtk\frame_{frame_number}.vtk", 
			frame_number = glob_wildcards(r"predictions\{label_1}_frame{frame_number}_{label_2}.tif")[1]
		),
		r"reports\figures\segmentation_rendered.mp4",
		r"reports\figures\segmentation_rendered.gif",
		#expand(r"data\interim\care\{label_1}_ch3_{label_2}.tif",
		#	label_1 = glob_wildcards(r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\{label_1}_ch1_{label_2}.tif")[0][:5],
		#	label_2 = glob_wildcards(r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\{label_1}_ch1_{label_2}.tif")[1][:5]),
		r"data\interim\care",
		r"..\2021_Iterative_Biofilm_Annotation\datasets\.eva-v1-dz400-care.chkpnt",
		r"models\eva-v1-dz400-care_rep1",
		r"data\interim\predictions\care\eva-v1-dz400-care_rep1\probs",
		r"data\interim\predictions\raw\eva-v1_dz400_rep1\probs"
		
		

rule create_vtk_from_labelfile:
	output:
		r"data\interim\vtk\frame_{frame_number}.vtk"
	input:
		r"predictions\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame{frame_number}_Nz54P.tif"
	shell:
		"""matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tif2vtk('{output}', '{input}')" """
		
rule create_rendering:
	output:
		r"reports\figures\segmentation_rendered.mp4"
	shell:
		r"C:\ffmpeg-3.4.2-win64-static\bin\ffmpeg.exe -pattern_type sequence -start_number 0 -framerate 15 -i data\interim\vtk_rendering\simple.%04d.png  -c:v libx264 reports\figures\segmentation_rendered.mp4"
		
		
rule mp4_to_gif:
	output:
		r"reports\figures\{filename}.gif"
	input:
		r"reports\figures\{filename}.mp4"
	conda:
		r"envs\ffmpeg.yml"
	shell:
		r"ffmpeg -i {input} -f gif {output}"
		
rule train_stardist_model:
	output:
		directory(r"models\{datasetname}_rep{rep_nummer}")
	input:
		r"..\2021_Iterative_Biofilm_Annotation\datasets\.{datasetname}.chkpnt"
	resources:
		nvidia_gpu=1
	conda:
		r"envs\stardist.yml"
	shell:
		r"python scripts\stardist_training.py {output} ..\2021_Iterative_Biofilm_Annotation\datasets\{wildcards.datasetname}"
		
		
rule stardist_prediction_probabilities:
	output:
		directory(r'data\interim\predictions\{data_folder}\{model_name}\probs')
	input:
		folder=r"Y:\Eric\prediction_test\data\interim\{data_folder}",
		model=r"models\{model_name}"
	params:
		output_dir=r"data\interim\predictions"
	resources:
		nvidia_gpu=1
	conda:
		r"envs\stardist.yml"
	shell:
		r"python scripts\stardist_prediction.py {input.folder} {input.model} {params.output_dir}\{wildcards.data_folder} --intp-factor 4 --probs"
		

