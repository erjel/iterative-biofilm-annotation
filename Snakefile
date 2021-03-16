#set CUDA_VISIBLE_DEVICES=0,1 && snakemake -j --use-conda --resources nvidia_gpu=2
#set CUDA_VISIBLE_DEVICES=1 && snakemake -j --use-conda --resources nvidia_gpu=1

include: r"workflows\snakefile_care"
include: r"workflows\snakefile_debug"

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
		r"..\2021_Iterative_Biofilm_Annotation_bk\datasets\.eva-v1-dz400-care.chkpnt",
		r"models\eva-v1-dz400-care_rep1",
		r"data\interim\predictions\care\eva-v1-dz400-care_rep1\.chkpnt",
		expand(r"data\interim\vtk\care\eva-v1-dz400-care_rep1\frame_{frame_number}.vtk", 
			frame_number = glob_wildcards(r"data\interim\predictions\care\eva-v1-dz400-care_rep1\{label_1}_frame{frame_number}_{label_2}.tif")[1]
		),
		r"reports\figures\care\eva-v1-dz400-care_rep1_segmentation_rendered.mp4",
		r"data\processed\tracks\care_model_eva-v1-dz400-care_rep1.csv",
		r"data\processed\tracks\care_model_eva-v1-dz400-care_rep1_growthrate.csv",
		r"data\interim\tracking\care_model_eva-v1-dz400-care_rep1.xml",
		r'reports\figures\care\eva-v1-dz400-care_rep1_single_cell_growthrate.png',
		r'reports\figures\care\eva-v1-dz400-care_rep1_growthrate_heatmap.png',
		r'reports\figures\care\BiofilmQ_single_cell_growthrate.png',
		r'reports\figures\care\BiofilmQ_growthrate_heatmap.png',
		expand(r"data\interim\vtk\care\BiofilmQ\frame_{frame_number}.vtk", 
			frame_number = glob_wildcards(r"data\interim\predictions\care\BiofilmQ\{label_1}_frame{frame_number}_{label_2}.tif")[1]
		),
		'data/interim/metadata/raw/z_standard_deviation.csv',
		'data/interim/training_sets/CARE_2D/raw_ch1_ch2/2021-02-03.npz',
		'models/care/2D_raw_ch1_ch2_2021-02-03_rep1',
		'data/interim/training_sets/CARE_3D/raw_ch1_ch2/2021-02-03.npz',
		'models/care/3D_raw_ch1_ch2_2021-02-03_rep1',
		'models/n2v/3D_raw_ch1_rep1',
		expand('data/interim/predictions/raw/n2v/n2v_3D_small/{label_1}_ch1_{label_2}.tif', 
			label_1 = glob_wildcards(r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\{label_1}_ch1_{label_2}.tif")[0],
			label_2 = glob_wildcards(r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\{label_1}_ch1_{label_2}.tif")[1],
		),
		'data/interim/predictions/raw/care/3D_raw_ch1_ch2_2021-02-03_rep1/.chkpnt',
		'data/interim/predictions/raw/care/2D_raw_ch1_ch2_2021-02-03_rep1/.chkpnt',
		'models/care/3D_raw_ch2_ch1_2020-01-05_rep1', # Uses the training data in "Y:\Eva\CARE\Created_Trainingdata\2020-01-05_ch2_ch1_testsplit.npz"
		'models/care/3D_raw_ch2_ch1_2020-01-05_rep2',
		'models/care/3D_raw_ch2_ch1_2020-01-05_rep3',
		'models/care/3D_raw_ch2_ch1_2020-01-05_rep4',
		'models/care/3D_raw_ch2_ch1_2020-01-05_rep5',
		#'G:/batch_decon/data/interim/huygens_parameterscan/huygens_batch_file_rep1.hgsb',
		#'data/interim/huygens_parameterscan',
		'data/processed/tracks/care_model_eva-v1-dz400-care_rep1_vtk',

rule create_vtks_with_track_id:
	output:
		directory('data/processed/tracks/{data}_model_{model}_vtk')
	input:
		tracks_csv = r"data\processed\tracks\{data}_model_{model}.csv",
		prediction_folder = r"data\interim\predictions\{data}\{model}",
	threads:
		1
	shell:
		"""matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tracks2vtk('{output}', '{input.tracks_csv}', '{input.prediction_folder}')" """
		
rule copy_huygens:
	output:
		directory('data/interim/huygens_parameterscan'),
	params:
		'G:/batch_decon/data/interim/huygens_parameterscan'
	threads:
		1
	shell:
		r"XCOPY {params} {output} /s /i"
		
rule huygens_parameterscan:
	output:
		'G:/batch_decon/data/interim/huygens_parameterscan/huygens_batch_file_rep1.hgsb',
	input:
		#r"Y:\Eva\CARE\08.12.20_19h\plasmid-100nm-19h\plasmid-100nm-19h_pos5_ch1_frame000001_Nz271.tif",
		r"Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame000083_Nz54.tif",
	params:
		prep_folder = 'G:/batch_decon/data/interim/huygens_parameterscan/prep_folder',
		result_folder = 'G:/batch_decon/data/interim/huygens_parameterscan/result_folder',
	threads:
		1
	shell:
		"""matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); run_huygens_parameter_scan('{output}',  '{input}', '{params.prep_folder}', '{params.result_folder}')" """

rule predict_n2v:
	output:
		'data/interim/predictions/raw/n2v/{model}/{tiff}.tif',
	params:
		model_path = r"\models\n2v\{model}",
		input_folder = r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5',
	resources:
		nvidia_gpu=1
	conda:
		'envs/n2v.yml'
	shell:
		r'python scripts/n2v_prediction.py {output} "{params.input_folder}\{wildcards.tiff}.tif" {params.model_path}' +\
			' --has_overview'

rule train_n2v:
	output:
		directory('models/n2v/3D_{data}_{ch}_rep{rep}')
	params:
		input_folder=r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5',
	resources:
		nvidia_gpu=1
	conda:
		'envs/n2v.yml'
	shell:
		r'python scripts/n2v_training.py {output} "{params.input_folder}" --seed {output}'

rule calc_coverslip_slice:
	output:
		std_csv = 'data/interim/metadata/{data}/z_standard_deviation.csv',
		cover_slip_slice_csv = 'data/interim/metadata/{data}/cover_slip_slice.csv',
	input:
		basepaths = ['Y:/Eva/CARE/08.12.20_14h', 'Y:/Eva/CARE/08.12.20_19h', 'Y:/Eva/CARE/08.12.20_24h', 'Y:/Eva/CARE/09.12.20_14h']
	conda:
		'envs/calc.yml'
	shell:
		'python scripts/calc_coverslip_slice.py {output.std_csv} {output.cover_slip_slice_csv} {input.basepaths}'

rule plot_growthrate_heatmap:
	output:
		r'reports\figures\{data}\{modelname}_growthrate_heatmap.png',
	input:
		r'data\processed\tracks\{data}_model_{modelname}_growthrate.csv',
	conda:
		r'envs\plot.yml'
	shell:
		r"python scripts\plot_growthrate_heatmap.py {output} {input}"

rule plot_growthrate:
	output:
		r'reports\figures\{data}\{modelname}_single_cell_growthrate.png',
	input:
		r'data\processed\tracks\{data}_model_{modelname}_growthrate.csv',
	conda:
		r'envs\calc.yml'
	shell:
		r"python scripts\plot_growthrate.py {output} {input}"

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
		r"envs\calc.yml"
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
		r"envs\calc.yml"
	shell:
		r"python scripts\calc_growthrate.py {output} {input.tracks_csv} {input.prediction_folder}"

rule biofilmQData2Labelimages:
	output:
		directory(r'Y:\Eric\prediction_test\data\interim\predictions\{data}\BiofilmQ'),
	params:
		input_folder = r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\data',
	threads:
		1
	shell:
		"""matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); data2labelimage('{output}',  '{params.trans}', '{output.input_folder}')" """


rule biofilmQ2trackmate:
	output:
		xml = r"data\interim\tracking\{data}_model_BiofilmQ.xml",
		trans = r"data\interim\tracking\{data}_model_BiofilmQ_translations.csv",
		crop = r"data\interim\tracking\{data}_model_BiofilmQ_crop_offsets.csv",
	input:
		data_folder = r'Y:\Daniel\000_Microscope data\2020.09.15_CNN3\kdv1502R_5L_30ms_300gain002\Pos5\data',
		int_data_path = r'data\interim\tracking\{data}.tif',
	threads:
		1
	shell:
		"""matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); biofilmQ2trackMate('{output.xml}',  '{output.trans}', '{output.crop}', '{input.int_data_path}', '{input.data_folder}')" """
	


		
rule labelimages2trackmate:
	output:
		r"data\interim\tracking\{data}_model_{model}.xml",
	input:
		int_data_path = r'data\interim\tracking\{data}.tif',
		input_folder = r'data\interim\predictions\{data}\{model}',
	conda:
		r"envs\jinja2.yml"
	shell:
		r"python scripts\labelimage2trackmate.py --int_data_path {input.int_data_path} --input_folder {input.input_folder} --output_xml {output}"
		
rule stack4trackmate:
	output:
		r"data\interim\tracking\{data}.tif"
	input:
		r"data\interim\{data}"
	conda:
		r"envs\stardist.yml"
	shell:
		r"python scripts\create_stack_for_trackmate.py {output} {input}"
				
		
		
rule trackmate2napari:
	output:
		r"data\processed\tracks\{data}_model_{model}.csv"
	input:
		r"data\interim\tracking\{data}_model_{model}_Tracks.xml" # comes from manual TrackMate step
	conda:
		r"envs\stardist.yml"
	shell:
		r"python scripts\trackmate_xml_to_napari_csv.py {input} {output}"
	
rule create_vtk_from_labelfile_test:
	output:
		r"data\interim\vtk\frame_{frame_number}.vtk"
	input:
		r"predictions\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame{frame_number}_Nz54P.tif"
	threads:
		2
	shell:
		"""matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tif2vtk('{output}', '{input}')" """
		
rule create_vtk_from_labelfile_general:
	output:
		r"data\interim\vtk\{data}\{model}\frame_{frame_number}.vtk"
	input:
		r"data\interim\predictions\{data}\{model}\kdv1502R_5L_30ms_300gain002_pos5_ch1_frame{frame_number}_Nz54.tif"
	threads:
		2
	shell:
		"""matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tif2vtk('{output}', '{input}')" """
		
		
rule create_vtk_from_labelfile_general_ch2:
	output:
		r"data\interim\vtk\{data}\{model}\frame_{frame_number}.vtk"
	input:
		r"data\interim\predictions\{data}\{model}\kdv1502R_5L_30ms_300gain002_pos5_ch2_frame{frame_number}_Nz54.tif"
	threads:
		2
	shell:
		"""matlab -nojvm -nosplash -batch "addpath(genpath('scripts')); tif2vtk('{output}', '{input}')" """

		
rule create_rendering_test:
	output:
		r"reports\figures\segmentation_rendered.mp4"
	shell:
		r"C:\ffmpeg-3.4.2-win64-static\bin\ffmpeg.exe -pattern_type sequence -start_number 0 -framerate 15 -i data\interim\vtk_rendering\simple.%04d.png  -c:v libx264 reports\figures\segmentation_rendered.mp4"
		
rule create_rendering_general:
	output:
		r"reports\figures\{data}\{model}_segmentation_rendered.mp4",
	conda:
		r"envs\ffmpeg.yml"
	shell:
		r"ffmpeg.exe -pattern_type sequence -start_number 0 -framerate 15 -i data\interim\vtk_rendering\{wildcards.data}\{wildcards.model}\frame.%04d.png  -c:v libx264 reports\figures\{wildcards.data}\{wildcards.model}_segmentation_rendered.mp4"
		
		

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
		r"..\2021_Iterative_Biofilm_Annotation_bk\datasets\.{datasetname}.chkpnt"
	resources:
		nvidia_gpu=1
	conda:
		r"envs\stardist.yml"
	shell:
		r"python scripts\stardist_training.py {output} ..\2021_Iterative_Biofilm_Annotation_bk\datasets\{wildcards.datasetname}"
		
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
		r"envs\stardist.yml"
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
		r"envs\stardist.yml"
	shell:
		r"python scripts\stardist_prediction.py {input.folder} {input.model} {params.output_dir}\{wildcards.data_folder} --intp-factor 4 --probs"
		

