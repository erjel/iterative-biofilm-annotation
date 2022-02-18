# trace generated using paraview version 5.8.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
import argparse


def main(input_filename, output_filename):

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'Legacy VTK Reader'
    a_full_biofilm_rawvtk = LegacyVTKReader(FileNames=[input_filename])

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1160, 944]

    # get layout
    layout1 = GetLayout()

    # show data in view
    a_full_biofilm_rawvtkDisplay = Show(a_full_biofilm_rawvtk, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'Area'
    areaLUT = GetColorTransferFunction('Area')
    areaLUT.RGBPoints = [1.0, 0.231373, 0.298039, 0.752941, 3795.5, 0.865003, 0.865003, 0.865003, 7590.0, 0.705882, 0.0156863, 0.14902]
    areaLUT.ScalarRangeInitialized = 1.0

    # trace defaults for the display properties.
    a_full_biofilm_rawvtkDisplay.Representation = 'Surface'
    a_full_biofilm_rawvtkDisplay.ColorArrayName = ['POINTS', 'Area']
    a_full_biofilm_rawvtkDisplay.LookupTable = areaLUT
    a_full_biofilm_rawvtkDisplay.OSPRayScaleArray = 'Area'
    a_full_biofilm_rawvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    a_full_biofilm_rawvtkDisplay.SelectOrientationVectors = 'Area'
    a_full_biofilm_rawvtkDisplay.ScaleFactor = 102.4
    a_full_biofilm_rawvtkDisplay.SelectScaleArray = 'Area'
    a_full_biofilm_rawvtkDisplay.GlyphType = 'Arrow'
    a_full_biofilm_rawvtkDisplay.GlyphTableIndexArray = 'Area'
    a_full_biofilm_rawvtkDisplay.GaussianRadius = 5.12
    a_full_biofilm_rawvtkDisplay.SetScaleArray = ['POINTS', 'Area']
    a_full_biofilm_rawvtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    a_full_biofilm_rawvtkDisplay.OpacityArray = ['POINTS', 'Area']
    a_full_biofilm_rawvtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    a_full_biofilm_rawvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    a_full_biofilm_rawvtkDisplay.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    a_full_biofilm_rawvtkDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    a_full_biofilm_rawvtkDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera()

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # show color bar/color legend
    a_full_biofilm_rawvtkDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get opacity transfer function/opacity map for 'Area'
    areaPWF = GetOpacityTransferFunction('Area')
    areaPWF.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]
    areaPWF.ScalarRangeInitialized = 1

    # destroy renderView1
    Delete(renderView1)
    del renderView1

    # load state
    LoadState('/u/ejelli/ptmp_link/PhD_thesis/chapter_deep_learning_segmentation/paraview/full_semimanual-raw_5um_scale.pvsm', DataDirectory='/u/ejelli/ptmp_link/PhD_thesis/chapter_deep_learning_segmentation/paraview',
        im0tifFileNames=['/u/ejelli/ptmp_link/PhD_thesis/chapter_single_cell_segmentation/full_semimanual-raw/test/images/im0.tif'])

    # find view
    renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [917, 782]

    # get layout
    layout1_1 = GetLayoutByName("Layout #1")

    # set active view
    SetActiveView(renderView1)

    # find source
    box1 = FindSource('Box1')

    # set active source
    SetActiveSource(box1)

    # find source
    im0tif = FindSource('im0.tif')

    # set active source
    SetActiveSource(im0tif)

    # get display properties
    im0tifDisplay = GetDisplayProperties(im0tif, view=renderView1)

    # toggle 3D widget visibility (only when running from the GUI)
    Show3DWidgets(proxy=im0tifDisplay)

    # toggle 3D widget visibility (only when running from the GUI)
    Hide3DWidgets(proxy=im0tifDisplay.SliceFunction)

    # toggle 3D widget visibility (only when running from the GUI)
    Hide3DWidgets(proxy=im0tifDisplay)
    
    # hide color bar/color legend
    im0tifDisplay.SetScalarBarVisibility(renderView1, False)

    # destroy im0tif
    Delete(im0tif)
    del im0tif

    # Properties modified on renderView1
    renderView1.EnableRayTracing = 0

    # set active source
    SetActiveSource(a_full_biofilm_rawvtk)

    # show data in view
    a_full_biofilm_rawvtkDisplay = Show(a_full_biofilm_rawvtk, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    a_full_biofilm_rawvtkDisplay.Representation = 'Surface'
    a_full_biofilm_rawvtkDisplay.ColorArrayName = ['POINTS', 'Area']
    a_full_biofilm_rawvtkDisplay.LookupTable = areaLUT
    a_full_biofilm_rawvtkDisplay.OSPRayScaleArray = 'Area'
    a_full_biofilm_rawvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    a_full_biofilm_rawvtkDisplay.SelectOrientationVectors = 'Area'
    a_full_biofilm_rawvtkDisplay.ScaleFactor = 102.4
    a_full_biofilm_rawvtkDisplay.SelectScaleArray = 'Area'
    a_full_biofilm_rawvtkDisplay.GlyphType = 'Arrow'
    a_full_biofilm_rawvtkDisplay.GlyphTableIndexArray = 'Area'
    a_full_biofilm_rawvtkDisplay.GaussianRadius = 5.12
    a_full_biofilm_rawvtkDisplay.SetScaleArray = ['POINTS', 'Area']
    a_full_biofilm_rawvtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    a_full_biofilm_rawvtkDisplay.OpacityArray = ['POINTS', 'Area']
    a_full_biofilm_rawvtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    a_full_biofilm_rawvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    a_full_biofilm_rawvtkDisplay.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    a_full_biofilm_rawvtkDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    a_full_biofilm_rawvtkDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]

    # show color bar/color legend
    a_full_biofilm_rawvtkDisplay.SetScalarBarVisibility(renderView1, True)

    #Enter preview mode
    layout1_1.PreviewMode = [1200, 1000]

    # set active view
    SetActiveView(None)

    # close an empty frame
    layout1_1.Collapse(2)

    # set active view
    SetActiveView(renderView1)

    #Exit preview mode
    layout1_1.PreviewMode = [0, 0]

    #Enter preview mode
    layout1_1.PreviewMode = [1200, 1000]

    # set scalar coloring
    ColorBy(a_full_biofilm_rawvtkDisplay, ('POINTS', 'RandomNumber'))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(areaLUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    a_full_biofilm_rawvtkDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    a_full_biofilm_rawvtkDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'RandomNumber'
    randomNumberLUT = GetColorTransferFunction('RandomNumber')
    randomNumberLUT.RGBPoints = [0.23106519877910614, 0.231373, 0.298039, 0.752941, 500.0118033513427, 0.865003, 0.865003, 0.865003, 999.7925415039062, 0.705882, 0.0156863, 0.14902]
    randomNumberLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'RandomNumber'
    randomNumberPWF = GetOpacityTransferFunction('RandomNumber')
    randomNumberPWF.Points = [0.23106519877910614, 0.0, 0.5, 0.0, 999.7925415039062, 1.0, 0.5, 0.0]
    randomNumberPWF.ScalarRangeInitialized = 1

    # hide color bar/color legend
    a_full_biofilm_rawvtkDisplay.SetScalarBarVisibility(renderView1, False)

    # Properties modified on a_full_biofilm_rawvtkDisplay
    a_full_biofilm_rawvtkDisplay.Scale = [0.63, 1.0, 1.0]

    # Properties modified on a_full_biofilm_rawvtkDisplay.DataAxesGrid
    a_full_biofilm_rawvtkDisplay.DataAxesGrid.Scale = [0.63, 1.0, 1.0]

    # Properties modified on a_full_biofilm_rawvtkDisplay.PolarAxes
    a_full_biofilm_rawvtkDisplay.PolarAxes.Scale = [0.63, 1.0, 1.0]

    # Properties modified on a_full_biofilm_rawvtkDisplay
    a_full_biofilm_rawvtkDisplay.Scale = [0.63, 0.63, 1.0]

    # Properties modified on a_full_biofilm_rawvtkDisplay.DataAxesGrid
    a_full_biofilm_rawvtkDisplay.DataAxesGrid.Scale = [0.63, 0.63, 1.0]

    # Properties modified on a_full_biofilm_rawvtkDisplay.PolarAxes
    a_full_biofilm_rawvtkDisplay.PolarAxes.Scale = [0.63, 0.63, 1.0]

    # Properties modified on renderView1
    renderView1.OrientationAxesVisibility = 0

    # Properties modified on renderView1
    renderView1.EnableRayTracing = 1

    # Properties modified on renderView1
    renderView1.BackEnd = 'OptiX pathtracer'

    # Properties modified on renderView1
    renderView1.LightScale = 0.6

    # Properties modified on renderView1
    renderView1.SamplesPerPixel = 50

    # current camera placement for renderView1
    renderView1.CameraPosition = [670.7552131950703, 1364.1816933345276, 740.760207000087]
    renderView1.CameraFocalPoint = [220.9628788358408, 129.482879638882, -197.9726006462275]
    renderView1.CameraViewUp = [-0.24048307553756215, -0.5304355957773788, 0.8128997288179474]
    renderView1.CameraParallelScale = 42.4352447854375

    # save screenshot
    SaveScreenshot(output_filename, layout1_1, ImageResolution=[2400, 2000])
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='INPUT', type=str)
    parser.add_argument('output', metavar='OUTPUT', type=str)
    args = parser.parse_args()
    
    print(args.input)
    print(args.output)
    main(args.input, args.output)
    
