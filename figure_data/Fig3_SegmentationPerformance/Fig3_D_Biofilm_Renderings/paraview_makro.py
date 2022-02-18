# trace generated using paraview version 5.8.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
import argparse


def main(input_filename, output_filename, field):

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'Legacy VTK Reader'
    biofilmQ_fnvtk = LegacyVTKReader(FileNames=[input_filename])

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1160, 944]

    # get layout
    layout1 = GetLayout()

    # show data in view
    biofilmQ_fnvtkDisplay = Show(biofilmQ_fnvtk, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'Area'
    areaLUT = GetColorTransferFunction('Area')
    areaLUT.RGBPoints = [1.0, 0.231373, 0.298039, 0.752941, 3795.5, 0.865003, 0.865003, 0.865003, 7590.0, 0.705882, 0.0156863, 0.14902]
    areaLUT.ScalarRangeInitialized = 1.0

    # trace defaults for the display properties.
    biofilmQ_fnvtkDisplay.Representation = 'Surface'
    biofilmQ_fnvtkDisplay.ColorArrayName = ['POINTS', 'Area']
    biofilmQ_fnvtkDisplay.LookupTable = areaLUT
    biofilmQ_fnvtkDisplay.OSPRayScaleArray = 'Area'
    biofilmQ_fnvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    biofilmQ_fnvtkDisplay.SelectOrientationVectors = 'Area'
    biofilmQ_fnvtkDisplay.ScaleFactor = 102.4
    biofilmQ_fnvtkDisplay.SelectScaleArray = 'Area'
    biofilmQ_fnvtkDisplay.GlyphType = 'Arrow'
    biofilmQ_fnvtkDisplay.GlyphTableIndexArray = 'Area'
    biofilmQ_fnvtkDisplay.GaussianRadius = 5.12
    biofilmQ_fnvtkDisplay.SetScaleArray = ['POINTS', 'Area']
    biofilmQ_fnvtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    biofilmQ_fnvtkDisplay.OpacityArray = ['POINTS', 'Area']
    biofilmQ_fnvtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    biofilmQ_fnvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    biofilmQ_fnvtkDisplay.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    biofilmQ_fnvtkDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    biofilmQ_fnvtkDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera()

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # show color bar/color legend
    biofilmQ_fnvtkDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get opacity transfer function/opacity map for 'Area'
    areaPWF = GetOpacityTransferFunction('Area')
    areaPWF.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]
    areaPWF.ScalarRangeInitialized = 1

    # set scalar coloring
    ColorBy(biofilmQ_fnvtkDisplay, ('POINTS', field))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(areaLUT, renderView1)

    # update the view to ensure updated data information
    renderView1.Update()

    # get opacity transfer function/opacity map for 'Area'
    areaPWF = GetOpacityTransferFunction('Area')
    areaPWF.Points = [1.0, 0.0, 0.5, 0.0, 7590.0, 1.0, 0.5, 0.0]
    areaPWF.ScalarRangeInitialized = 1

    # set scalar coloring
    ColorBy(biofilmQ_fnvtkDisplay, ('POINTS', field))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(areaLUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    biofilmQ_fnvtkDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    biofilmQ_fnvtkDisplay.SetScalarBarVisibility(renderView1, False)


    # get color transfer function/color map for 'biofilmQ_fn'
    biofilmQ_fnLUT = GetColorTransferFunction(field)
    biofilmQ_fnLUT.RGBPoints = [1.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 2.0, 0.705882, 0.0156863, 0.14902]
    biofilmQ_fnLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'biofilmQ_fn'
    biofilmQ_fnPWF = GetOpacityTransferFunction(field)
    biofilmQ_fnPWF.Points = [1.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]
    biofilmQ_fnPWF.ScalarRangeInitialized = 1

    # Properties modified on biofilmQ_fnvtkDisplay
    biofilmQ_fnvtkDisplay.Scale = [0.63, 1.0, 1.0]

    # Properties modified on biofilmQ_fnvtkDisplay.DataAxesGrid
    biofilmQ_fnvtkDisplay.DataAxesGrid.Scale = [0.63, 1.0, 1.0]

    # Properties modified on biofilmQ_fnvtkDisplay.PolarAxes
    biofilmQ_fnvtkDisplay.PolarAxes.Scale = [0.63, 1.0, 1.0]

    # Properties modified on biofilmQ_fnvtkDisplay
    biofilmQ_fnvtkDisplay.Scale = [0.63, 0.63, 1.0]

    # Properties modified on biofilmQ_fnvtkDisplay.DataAxesGrid
    biofilmQ_fnvtkDisplay.DataAxesGrid.Scale = [0.63, 0.63, 1.0]

    # Properties modified on biofilmQ_fnvtkDisplay.PolarAxes
    biofilmQ_fnvtkDisplay.PolarAxes.Scale = [0.63, 0.63, 1.0]

    # create a new 'Box'
    box2 = Box()

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1505, 793]

    # get layout
    layout1 = GetLayout()

    # Properties modified on renderView1
    renderView1.Background = [1.0, 1.0, 1.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # find source
    box1 = FindSource('Box1')

    # find source
    plane1 = FindSource('Plane1')

    # Properties modified on box2
    box2.XLength = 50.0
    box2.YLength = 5.0
    box2.ZLength = 5.0
    box2.Center = [450.0, 680.0, 20.0]

    # show data in view
    box2Display = Show(box2, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    box2Display.Representation = 'Surface'
    box2Display.ColorArrayName = [None, '']
    box2Display.OSPRayScaleArray = 'Normals'
    box2Display.OSPRayScaleFunction = 'PiecewiseFunction'
    box2Display.SelectOrientationVectors = 'None'
    box2Display.ScaleFactor = 5.0
    box2Display.SelectScaleArray = 'None'
    box2Display.GlyphType = 'Arrow'
    box2Display.GlyphTableIndexArray = 'None'
    box2Display.GaussianRadius = 0.25
    box2Display.SetScaleArray = ['POINTS', 'Normals']
    box2Display.ScaleTransferFunction = 'PiecewiseFunction'
    box2Display.OpacityArray = ['POINTS', 'Normals']
    box2Display.OpacityTransferFunction = 'PiecewiseFunction'
    box2Display.DataAxesGrid = 'GridAxesRepresentation'
    box2Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    box2Display.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    box2Display.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # update the view to ensure updated data information
    renderView1.Update()

    # get color transfer function/color map for 'RandomNumber'
    randomNumberLUT = GetColorTransferFunction('RandomNumber')
    randomNumberLUT.RGBPoints = [0.23106519877910614, 0.231373, 0.298039, 0.752941, 500.0118033513427, 0.865003, 0.865003, 0.865003, 999.7925415039062, 0.705882, 0.0156863, 0.14902]
    randomNumberLUT.ScalarRangeInitialized = 1.0

    # Rescale transfer function
    randomNumberLUT.RescaleTransferFunction(0.07026065140962601, 999.7925415039062)

    # get opacity transfer function/opacity map for 'RandomNumber'
    randomNumberPWF = GetOpacityTransferFunction('RandomNumber')
    randomNumberPWF.Points = [0.23106519877910614, 0.0, 0.5, 0.0, 999.7925415039062, 1.0, 0.5, 0.0]
    randomNumberPWF.ScalarRangeInitialized = 1

    # Rescale transfer function
    randomNumberPWF.RescaleTransferFunction(0.07026065140962601, 999.7925415039062)

    # create a new 'Plane'
    plane2 = Plane()

    # Properties modified on plane2
    plane2.Origin = [-20000.0, -20000.0, 10.0]
    plane2.Point1 = [1200.0, -20000.0, 10.0]
    plane2.Point2 = [-20000.0, 1200.0, 10.0]

    # show data in view
    plane2Display = Show(plane2, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    plane2Display.Representation = 'Surface'
    plane2Display.ColorArrayName = [None, '']
    plane2Display.OSPRayScaleArray = 'Normals'
    plane2Display.OSPRayScaleFunction = 'PiecewiseFunction'
    plane2Display.SelectOrientationVectors = 'None'
    plane2Display.ScaleFactor = 2120.0
    plane2Display.SelectScaleArray = 'None'
    plane2Display.GlyphType = 'Arrow'
    plane2Display.GlyphTableIndexArray = 'None'
    plane2Display.GaussianRadius = 106.0
    plane2Display.SetScaleArray = ['POINTS', 'Normals']
    plane2Display.ScaleTransferFunction = 'PiecewiseFunction'
    plane2Display.OpacityArray = ['POINTS', 'Normals']
    plane2Display.OpacityTransferFunction = 'PiecewiseFunction'
    plane2Display.DataAxesGrid = 'GridAxesRepresentation'
    plane2Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    plane2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    plane2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # update the view to ensure updated data information
    renderView1.Update()

    # set active source
    SetActiveSource(box2)

    # change solid color
    box2Display.AmbientColor = [0.0, 0.0, 0.0]
    box2Display.DiffuseColor = [0.0, 0.0, 0.0]
    
    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    biofilmQ_fnLUT.ApplyPreset('Preset', True)

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    biofilmQ_fnPWF.ApplyPreset('Preset', True)

    # Rescale transfer function
    biofilmQ_fnLUT.RescaleTransferFunction(0.0, 2.0)

    # Rescale transfer function
    biofilmQ_fnPWF.RescaleTransferFunction(0.0, 2.0)

    
    # Properties modified on renderView1
    renderView1.EnableRayTracing = 1
    
    # Properties modified on renderView1
    renderView1.BackEnd = 'OptiX pathtracer'

    # Properties modified on renderView1
    renderView1.LightScale = 0.01

    # Properties modified on renderView1
    renderView1.SamplesPerPixel = 50
    
    
    # Properties modified on renderView1
    renderView1.OrientationAxesVisibility = 0

    # current camera placement for renderView1
    renderView1.CameraPosition = [670.7552131950703, 1364.1816933345276, 740.760207000087]
    renderView1.CameraFocalPoint = [220.9628788358408, 129.482879638882, -197.9726006462275]
    renderView1.CameraViewUp = [-0.24048307553756215, -0.5304355957773788, 0.8128997288179474]
    renderView1.CameraParallelScale = 42.4352447854375
    


    # save screenshot
    SaveScreenshot(output_filename, renderView1, ImageResolution=[2400, 2000])
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='INPUT', type=str)
    parser.add_argument('output', metavar='OUTPUT', type=str)
    args = parser.parse_args()
    
    print(args.input)
    print(args.output)
    field = args.input.split('/')[-1].split('.')[0]
    print(field)
    main(args.input, args.output, field)
    
