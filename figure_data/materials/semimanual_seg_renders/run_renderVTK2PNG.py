from subprocess import run
from pathlib import Path

vtk_files = sorted(Path('.').glob('*.vtk'))

for vtk_file in vtk_files:
    run(['vglrun', 'pvpython', 'paraview_vtk2png.py', str(vtk_file), vtk_file.stem + '_segmentation.png'])