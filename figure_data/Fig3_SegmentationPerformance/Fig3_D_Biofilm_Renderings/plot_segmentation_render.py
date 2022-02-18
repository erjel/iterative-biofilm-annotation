from subprocess import run
from pathlib import Path

vtk_files = sorted(Path('.').glob('*.vtk'))
is_cluster_node = False

for vtk_file in vtk_files:
	if is_cluster_node:
		run(['vglrun', 'pvpython', 'paraview_makro.py', str(vtk_file), vtk_file.stem + '_segmentation.png'])
	else:
		ray_samples = 50
		pv_python = r"C:\Program Files\ParaView 5.8.0-Windows-Python3.7-msvc2015-64bit\bin\pvpython.exe"
		run([pv_python, 'template_without_ray_tracing_main.py', str(vtk_file), vtk_file.stem + '_segmentation.png', '--ray_samples', str(ray_samples)])