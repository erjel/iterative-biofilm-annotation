# trace generated using paraview version 5.8.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from argparse import ArgumentParser
from pathlib import Path
from paraview.simple import *


def main(input_filename: Path, output_filename: Path, field: str, state_file: Path) -> None:
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1612, 806]

    # get layout
    layout1 = GetLayout()

    # destroy renderView1
    Delete(renderView1)
    del renderView1

    # load state
    LoadState(str(state_file), LoadStateDataFileOptions='Choose File Names',
        DataDirectory=str(state_file.parent),
        biofilmQ_fpvtkFileNames=[str(input_filename)])

    # find view
    renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1854, 1692]

    # get layout
    layout1_1 = GetLayoutByName("Layout #1")

    # set active view
    SetActiveView(renderView1)

    # Properties modified on renderView1
    renderView1.EnableRayTracing = 0

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # find source
    biofilmq_fpvtk = FindSource(input_filename.name)

    # set active source
    SetActiveSource(biofilmq_fpvtk)

    # get color transfer function/color map for 'biofilmQ_fp'
    biofilmQ_fpLUT = GetColorTransferFunction(field)
    biofilmQ_fpLUT.RGBPoints = [1.0, 1.0, 0.0, 1.0, 1.5, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]
    biofilmQ_fpLUT.ColorSpace = 'Step'
    biofilmQ_fpLUT.NanColor = [0.803922, 0.0, 0.803922]
    biofilmQ_fpLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'biofilmQ_fp'
    biofilmQ_fpPWF = GetOpacityTransferFunction(field)
    biofilmQ_fpPWF.Points = [1.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]
    biofilmQ_fpPWF.ScalarRangeInitialized = 1

    # get display properties
    biofilmq_fpvtkDisplay = GetDisplayProperties(biofilmq_fpvtk, view=renderView1)

    # set scalar coloring
    ColorBy(biofilmq_fpvtkDisplay, ('POINTS', field))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(biofilmQ_fpLUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    biofilmq_fpvtkDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    biofilmq_fpvtkDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'biofilmq_fp'
    biofilmq_fpLUT = GetColorTransferFunction(field)
    biofilmq_fpLUT.RGBPoints = [1.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 2.0, 0.705882, 0.0156863, 0.14902]
    biofilmq_fpLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'biofilmq_fp'
    biofilmq_fpPWF = GetOpacityTransferFunction(field)
    biofilmq_fpPWF.Points = [1.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]
    biofilmq_fpPWF.ScalarRangeInitialized = 1

    # Properties modified on biofilmq_fpLUT
    biofilmq_fpLUT.RGBPoints = [1.0, 1.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0]

    # Properties modified on renderView1
    renderView1.EnableRayTracing = 1

    # Properties modified on renderView1
    renderView1.BackEnd = 'OptiX pathtracer'

    # Properties modified on renderView1
    renderView1.AmbientSamples = 5

    # Properties modified on renderView1
    renderView1.SamplesPerPixel = 100

    # current camera placement for renderView1
    renderView1.CameraPosition = [1082.3800862491917, 1106.2122258924596, 635.2887899184091]
    renderView1.CameraFocalPoint = [341.8735350167798, 365.70567466004934, 30.66772268256308]
    renderView1.CameraViewUp = [-0.3535533905932741, -0.3535533905932738, 0.8660254037844386]
    renderView1.CameraParallelScale = 458.2265430984984

    biofilmq_fpvtkDisplay.SetScalarBarVisibility(renderView1, False)

    # save screenshot
    SaveScreenshot(str(output_filename), renderView1, ImageResolution=[2400, 2000])

    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input', metavar='INPUT', type=Path)
    parser.add_argument('output', metavar='OUTPUT', type=Path)
    parser.add_argument('state', type=Path)
    args = parser.parse_args()
    
    print(args.input)
    print(args.output)
    field = args.input.stem
    print(field)
    main(args.input, args.output, field, args.state)