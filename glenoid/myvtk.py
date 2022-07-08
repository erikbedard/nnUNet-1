import pyvista
from vtkmodules.all import *

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtk.util.numpy_support import *
import numpy as np
import copy
import statistics
import pyacvd

def get_rgba(color, a=1):
    if type(color) is not str and len(color) is 4:
        a = color[3]

    rgb = get_rgb(color)
    rgba = rgb.copy()
    rgba.append(a)
    return rgba


def get_rgb(color):
    # input is either RGB triplet or named color
    # output is an RGB triplet
    #
    # see list of acceptable color names here:
    # https://htmlpreview.github.io/?https://github.com/Kitware/vtk-examples/blob/gh-pages/VTKNamedColorPatches.html

    if type(color) is str:
        colors = vtkNamedColors()
        rgb = list(colors.GetColor3d(color))
    else:
        rgb = list(color)[0:3]
    return rgb


def plt_point(renderer: vtkRenderer, p, radius=1.0, color='black'):
    """
    Add a point to a scene.
    Args:
        renderer:
        p:
        radius:
        color:
    """
    point = vtkSphereSource()
    point.SetCenter(p)
    point.SetRadius(radius)
    point.SetPhiResolution(100)
    point.SetThetaResolution(100)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(point.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(get_rgb(color))

    renderer.AddActor(actor)


def plt_line(renderer: vtkRenderer, p1, p2, color='blue'):
    """
    Add a line to a scene.
    Args:
        renderer:
        p1:
        p2:
        color:
    """
    line = vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(get_rgb(color))

    renderer.AddActor(actor)


def plt_plane(renderer: vtkRenderer, plane, pt1, pt2, color='LightBlue', opacity=0.5):
    """
    Add a plane to a scene.
    Args:
        renderer:
        plane:
        color:
    """
    plane_source = vtkPlaneSource()
    plane_source.SetCenter(plane.GetOrigin())
    plane_source.SetNormal(plane.GetNormal())
    plane_source.SetPoint1(pt1)
    plane_source.SetPoint2(pt2)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(plane_source.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(get_rgb(color))
    actor.GetProperty().SetOpacity(opacity)

    renderer.AddActor(actor)


def plt_polydata(renderer: vtkRenderer, polydata: vtkPolyData, color='tomato', show_edges=False, edge_color='black'):
    """
    Add polydata to a scene.
    Args:
        renderer:
        polydata:
        color:
        show_edges:
        edge_color:
    """
    rgb = get_rgb(color)
    rgba = get_rgba(color)
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # handle scalar data
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(1)
    lut.SetTableValue(0, rgba)
    lut.Build()
    mapper.SetScalarRange(1, 1)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToMapScalars()
    mapper.SetLookupTable(lut)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(rgb)
    actor.GetProperty().SetOpacity(rgba[3])
    if show_edges is True:
        actor.GetProperty().SetEdgeVisibility(1)
        edges_rgb = get_rgb(edge_color)
        actor.GetProperty().SetEdgeColor(edges_rgb)

    renderer.AddActor(actor)


def show_scene(renderer: vtkRenderer, bg_color='white', window_name='VTK', window_size=(500, 500), save_path=None):
    """
    Create and render a scene.
    Args:
        renderer:
        bg_color:
        window_name:
        window_size:
    """
    renderer.SetBackground(get_rgb(bg_color))

    renWin = vtkRenderWindow()
    renWin.AddRenderer(renderer)
    renWin.SetSize(*window_size)
    renWin.SetWindowName(window_name)
    if save_path is not None:
        renWin.ShowWindowOff()
    renWin.Render()

    if save_path is not None:
        save_window(renWin, save_path)
    else:
        iren = vtkRenderWindowInteractor()
        style = vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
        iren.SetRenderWindow(renWin)
        iren.Initialize()
        iren.Start()


def save_window(renWin: vtkRenderWindow, fileName, rgba=False):
    '''
    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param fileName: The file name, if no extension then PNG is assumed.
    :param renWin: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    '''

    import os

    if fileName:
        # Select the writer to use.
        path, ext = os.path.splitext(fileName)
        ext = ext.lower()
        if not ext:
            ext = '.png'
            fileName = fileName + ext
        if ext == '.bmp':
            writer = vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtkPostScriptWriter()
        elif ext == '.tiff':
            writer = vtkTIFFWriter()
        else:
            writer = vtkPNGWriter()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(renWin)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            windowto_image_filter.Update()

        writer.SetFileName(fileName)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')


def get_mask_from_labels(image: vtkImageData, label_num: int):
    """
    Extracts a binary mask from an image with labels.
    Args:
        image: image with integer label data (e.g. 0, 1, 2...)
        label_num: number of the label to be extracted
    Returns:
        vtkImageData object with binary mask of selected label
    """
    threshold = vtkImageThreshold()
    threshold.SetInputData(image)
    threshold.ThresholdBetween(label_num, label_num)
    threshold.SetInValue(1)
    threshold.SetOutValue(0)
    threshold.Update()
    return threshold.GetOutput()


def calc_curvature(poly: vtkPolyData, method='mean'):
    curves = vtkCurvatures()
    curves.SetInputData(poly)

    available_curves = ['gaussian', 'mean', 'max', 'min']
    if method is 'gaussian':
        curves.SetCurvatureTypeToGaussian()

    elif method is 'mean':
        curves.SetCurvatureTypeToMean()

    elif method is 'max':
        curves.SetCurvatureTypeToMaximum()

    elif method is 'min':
        curves.SetCurvatureTypeToMinimum()

    else:
        return RuntimeError

    curves.Update()
    return curves.GetOutput()


def get_foreground_from_labels(poly: vtkPolyData, label_num: int):
    """
    Extracts foreground from poly data with labels.
    Args:
        poly: polydata with integer label data (e.g. 0, 1, 2...)
        label_num: number of the label to be extracted
    Returns:
         vtkUnstructuredGrid object with foreground data of selected label
    """
    threshold = vtkThreshold()
    threshold.SetInputData(poly)
    threshold.SetLowerThreshold(label_num)
    threshold.SetUpperThreshold(label_num)
    threshold.Update()
    return threshold.GetOutput()


def convert_image_to_numpy(image: vtkImageData):
    rows, cols, _ = image.GetDimensions()
    values = image.GetPointData().GetScalars()
    np_image = vtk_to_numpy(values)
    np_image = np_image.reshape(rows, cols, -1)
    return np_image


def ray_casting(poly: vtkPolyData, point_source, point_target):
    # Based on the following example:
    # Ray Casting with Python and VTK: Intersecting lines/rays with surface meshes
    # https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/

    # initialize oriented bounding box
    obbTree = vtkOBBTree()
    obbTree.SetDataSet(poly)
    obbTree.BuildLocator()

    # get intersection points
    pointsVTKintersection = vtkPoints()
    obbTree.IntersectWithLine(point_target, point_source, pointsVTKintersection, None)

    # extract intersection points
    points_itersection_data = pointsVTKintersection.GetData()
    no_points_intersection = points_itersection_data.GetNumberOfTuples()
    points_ntersection = []
    for idx in range(no_points_intersection):
        pts_tuple = points_itersection_data.GetTuple3(idx)
        points_ntersection.append(pts_tuple)

    return np.asarray(points_ntersection)


def read_nifti(filepath):
    """
    Read a nifti image (.nii or .nii.gz).
    Args:
        filepath: path of NIFTI image to be read
    Returns:
        vtkImageData object
    """
    reader = vtkNIFTIImageReader()
    reader.SetFileName(filepath)
    reader.Update()
    vtk_image = reader.GetOutput()
    return vtk_image


def read_stl(filepath):
    """
    Read a '.stl' file.
    Args:
        filepath: path of stl file to be read
    Returns:
        vtkPolyData object
    """
    reader = vtkSTLReader()
    reader.SetFileName(filepath)
    # 'update' the reader i.e. read the .stl file
    reader.Update()

    mesh = reader.GetOutput()

    # If there are no points in 'vtkPolyData' something went wrong
    if mesh.GetNumberOfPoints() == 0:
        raise ValueError(
            "No point data could be loaded from '" + filepath)

    return mesh


def save_stl(poly: vtkPolyData, save_path):
    """
    Save triangulation data to file.
    Args:
        save_path: directory where file will be saved (be sure to include the .stl extension)
        poly: vtkPolyData object with triangulation data to be saved
    """
    writer = vtkSTLWriter()
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    writer.SetFileName(save_path)
    writer.Write()


def save_nifti(image: vtkImageData, save_path):
    """
    Save imagedata data to file.
    Args:
        save_path: directory where file will be saved (be sure to include the nii.gz extension)
        poly: vtkPolyData object with triangulation data to be saved
    """
    writer = vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(save_path)
    writer.Write()


def decimate_polydata(poly: vtkPolyData, target=0.9, force=False):
    # Mesh decimation example:
    # https://kitware.github.io/vtk-examples/site/Python/Meshes/Decimation/
    """
    Down-sample (decimate) polydata with vtkDecimatePro.
    Args:
        poly:
        target:
            if 0 < target < 1, then this value represents a target reduction, e.g. 0.9 -> 90% reduction
            if target > 1, then this value represents the target number of triangles in the mesh
    Returns:
        decimated vtkPolyData object
    """
    if target < 1:
        reduction = target
    else:
        num_cells = poly.GetNumberOfCells()
        if target > num_cells:
            return RuntimeError  # target too high
        reduction = 1 - target / num_cells

    decimate = vtkDecimatePro()
    decimate.PreserveTopologyOn()
    if force is True:
        decimate.PreserveTopologyOff()
        decimate.SplittingOn()
        decimate.BoundaryVertexDeletionOn()
        decimate.SetMaximumError(VTK_DOUBLE_MAX)
    decimate.SetInputData(poly)
    decimate.SetTargetReduction(reduction)

    decimate.Update()
    return decimate.GetOutput()


def smooth_polydata(poly: vtkPolyData, n_iterations=15, pass_band=0.001, feature_angle=120.0):
    # Marching cubes + smoothing example:
    # https://kitware.github.io/vtk-examples/site/Python/Visualization/FrogBrain/
    """
    Smooth polydata with vtkWindowedSincPolyDataFilter.
    Args:
        poly:
        n_iterations:
        pass_band:
        feature_angle:
    Returns:
        smoothed vtkPolyData object
    """
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(n_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def convert_voxels_to_poly(binary_mask: vtkImageData, method='flying_edges'):
    # Surface Extraction: Creating a mesh from pixel-data using Python and VTK
    # https://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/

    # Marching cubes + smoothing example:
    # https://kitware.github.io/vtk-examples/site/Python/Visualization/FrogBrain/

    valid_methods = ['flying_edges', 'marching_cubes', 'preserve_boundary']
    if method is 'flying_edges' or method is 'marching_cubes':
        if method is 'flying_edges':
            surface = vtkDiscreteFlyingEdges3D()
        elif method is 'marching_cubes':
            surface = vtkDiscreteMarchingCubes()
        else:
            return RuntimeError

        surface.SetInputData(binary_mask)
        surface.GenerateValues(1, 1, 1)
        surface.ComputeScalarsOff()
        surface.ComputeNormalsOff()
        surface.ComputeGradientsOff()
        surface.Update()
        poly = surface.GetOutput()

    elif method is 'preserve_boundary':
        foreground_label = 1
        poly = convert_voxels_to_cube_mesh(binary_mask, foreground_label)
    else:
        return RuntimeError

    return poly


def get_points(source: vtkPolyData, point_ids=None):
    if point_ids is None:
        point_ids = np.arange(0, source.GetNumberOfPoints())

    points = []
    for p in point_ids:
        points.append(source.GetPoint(p))

    return np.asarray(points)


def subdivide_mesh(source: vtkPolyData):
    filter = vtkLoopSubdivisionFilter()
    filter.SetInputData(source)
    filter.Update()
    return filter.GetOutput()


def extract_mesh_data(triangle_mesh: vtkPolyData):
    """
    Args:
        triangle_mesh:
    Returns:
        vertices, faces as np arrays

    """
    # process faces
    face_list = []
    num_faces = triangle_mesh.GetNumberOfCells()
    for i in range(0, num_faces):
        cell = triangle_mesh.GetCell(i)

        vert_id_list = []
        for j in range(0, 3):
            vert_id = cell.GetPointId(j)
            vert_id_list.append(vert_id)

        face_list.append(vert_id_list)

    # process vertices
    points = get_points(triangle_mesh)

    return np.asarray(points), np.asarray(face_list)


def surface_from_points(points, bins=256):
    # code copied from vedo
    """Surface reconstruction from a scattered cloud of points.
    :param int bins: number of voxels in x, y and z.
    .. hint:: |recosurface| |recosurface.py|_
    """
    N = len(points)
    if N < 50:
        print("recoSurface: Use at least 50 points.")
        # return None
    points = np.array(points)

    ptsSource = vtkPointSource()
    ptsSource.SetNumberOfPoints(N)
    ptsSource.Update()
    vpts = ptsSource.GetOutput().GetPoints()
    for i, p in enumerate(points):
        vpts.SetPoint(i, p)
    polyData = ptsSource.GetOutput()

    distance = vtkSignedDistance()
    f = 0.1
    x0, x1, y0, y1, z0, z1 = polyData.GetBounds()
    distance.SetBounds(x0-(x1-x0)*f, x1+(x1-x0)*f,
                       y0-(y1-y0)*f, y1+(y1-y0)*f,
                       z0-(z1-z0)*f, z1+(z1-z0)*f)
    if polyData.GetPointData().GetNormals():
        distance.SetInputData(polyData)
    else:
        normals = vtkPCANormalEstimation()
        normals.SetInputData(polyData)
        normals.SetSampleSize(int(N / 50))
        normals.SetNormalOrientationToGraphTraversal()
        distance.SetInputConnection(normals.GetOutputPort())
        print("Recalculating normals for", N, "Points, sample size=", int(N / 50))

    b = polyData.GetBounds()
    diagsize = np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
    radius = diagsize / bins * 5
    distance.SetRadius(radius)
    distance.SetDimensions(bins, bins, bins)
    distance.Update()

    print("Calculating mesh from points with R =", radius)
    surface = vtkExtractSurface()
    surface.SetRadius(radius * 0.99)
    surface.HoleFillingOn()
    surface.ComputeNormalsOff()
    surface.ComputeGradientsOff()
    surface.SetInputConnection(distance.GetOutputPort())
    surface.Update()

    return surface.GetOutput()


def create_blank_image_from_source(source: vtkImageData, value=1):
    spacing = source.GetSpacing()
    dim = source.GetDimensions()
    origin = source.GetOrigin()

    image = vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(dim)
    image.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
    image.SetOrigin(origin)
    image.AllocateScalars(VTK_UNSIGNED_CHAR, 0)

    set_scalars(image, value)

    return image


def polydata_to_imagedata(polydata, image_template: vtkImageData=None, dimensions=(100, 100, 100), padding=1):
    # based on https://github.com/tfmoraes/polydata_to_imagedata/blob/main/polydata_to_imagedata.py
    inval = 1
    outval = 0
    if image_template is None:
        xi, xf, yi, yf, zi, zf = polydata.GetBounds()
        dx, dy, dz = dimensions

        # Calculating spacing
        sx = (xf - xi) / dx
        sy = (yf - yi) / dy
        sz = (zf - zi) / dz
        spacing = (sx,sy,sz)

        # Calculating Origin
        ox = xi + sx / 2.0
        oy = yi + sy / 2.0
        oz = zi + sz / 2.0
        origin = (ox,oy,oz)

        if padding:
            ox -= sx
            oy -= sy
            oz -= sz

            dx += 2 * padding
            dy += 2 * padding
            dz += 2 * padding
            dim = (dx,dy,dz)

        image_template = vtkImageData()
        image_template.SetSpacing(spacing)
        image_template.SetDimensions(dim)
        image_template.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
        image_template.SetOrigin(origin)
        image_template.AllocateScalars(VTK_UNSIGNED_CHAR, 0)

    origin = image_template.GetOrigin()
    spacing = image_template.GetSpacing()

    image = image_template


    # for i in range(image.GetNumberOfPoints()):
    #     image.GetPointData().GetScalars().SetTuple1(i, inval)

    pol2stenc = vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtkImageStencil()
    imgstenc.SetInputData(image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    return imgstenc.GetOutput()

def extract_image_geometry_from_mesh(image: vtkImageData, mesh: vtkPolyData):
    implicit = vtkImplicitPolyDataDistance()
    implicit.SetInput(mesh)

    extract = vtkExtractGeometry()
    extract.SetInputData(image)
    extract.SetImplicitFunction(implicit)
    extract.Update()
    return extract.GetOutput()


def find_closest_point(poly: vtkPolyData, point):
    """
    Find point on polydata that is closest to the query point.
    Args:
        point: query point
        poly:

    Returns:
        closest_point: point coordinates
        distance: signed distance from surface to query (-ve inside, +ve outside)
        point_id: ID of the closest_point
    """
    locator = vtkKdTreePointLocator()  # can also use vtkPointLocator (slower)
    locator.SetDataSet(poly)
    point_id = locator.FindClosestPoint(point)
    closest_point = np.asarray(poly.GetPoint(point_id))
    distance = np.linalg.norm(point-closest_point, 2)

    # get sign
    implicit_function = vtkImplicitPolyDataDistance()
    implicit_function.SetInput(poly)
    if implicit_function.FunctionValue(point) <= 0:
        distance = -distance  # inside
    else:
        pass  # outside
    return closest_point, distance, point_id


def clean_polydata(poly: vtkPolyData):
    clean = vtkCleanPolyData()
    clean.setInputData(poly)
    clean.Update()
    return clean.GetOutput()


def create_pointcloud_polydata(points, colors=None):

    """https://github.com/lmb-freiburg/demon
    Creates a vtkPolyData object with the point cloud from numpy arrays

    points: numpy.ndarray
        pointcloud with shape (n,3)

    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkPolyData object
    """
    vpoints = vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i,:])
    vpoly = vtkPolyData()
    vpoly.SetPoints(vpoints)

    if not colors is None:
        vcolors = vtkUnsignedCharArray()
        vcolors.SetNumberOfComponents(3)
        vcolors.SetName("Colors")
        vcolors.SetNumberOfTuples(points.shape[0])
        for i in range(points.shape[0]):
            vcolors.SetTuple3(i, colors[i, 0], colors[i, 1], colors[i, 2])
        vpoly.GetPointData().SetScalars(vcolors)

    vcells = vtkCellArray()

    for i in range(points.shape[0]):
        vcells.InsertNextCell(1)
        vcells.InsertCellPoint(i)

    vpoly.SetVerts(vcells)

    return vpoly


def convert_voxels_to_cube_mesh(image: vtkImageData, label_num: int):
    """
    Extracts a binary mask from an image label and converts it into a triangulated
    surface mesh which preserves the original cubic shape of the label's voxels.
    Args:
        image: image with integer labels
        label_num: the label which will be extracted

    Returns:
        vtkPolyData object with triangulated mesh

    """
    # Based on:
    # https://kitware.github.io/vtk-examples/site/Python/Medical/GenerateCubesFromLabels/

    # Pad the volume so that we can change the point data into cell
    # data.
    extent = image.GetExtent()
    pad = vtkImageWrapPad()
    pad.SetInputData(image)
    pad.SetOutputWholeExtent(extent[0], extent[1] + 1, extent[2], extent[3] + 1, extent[4], extent[5] + 1)
    pad.Update()

    # Copy the scalar point data of the volume into the scalar cell data
    pad.GetOutput().GetCellData().SetScalars(image.GetPointData().GetScalars())

    selector = vtkThreshold()
    selector.SetInputArrayToProcess(0, 0, 0, vtkDataObject().FIELD_ASSOCIATION_CELLS,
                                    vtkDataSetAttributes().SCALARS)
    selector.SetInputConnection(pad.GetOutputPort())
    selector.SetLowerThreshold(label_num)
    selector.SetUpperThreshold(label_num)
    selector.Update()

    # # Shift the geometry by 1/2
    # transform = vtkTransform()
    # transform.Translate(-0.5, -0.5, -0.5)
    #
    # transform_model = vtkTransformFilter()
    # transform_model.SetTransform(transform)
    # transform_model.SetInputConnection(selector.GetOutputPort())
    transform_model = selector

    geometry = vtkGeometryFilter()
    geometry.SetInputConnection(transform_model.GetOutputPort())
    geometry.Update()

    # convert into triangulated mesh for ease of further processing (e.g. smooth filter)
    trifilter = vtkTriangleFilter()
    trifilter.SetInputData(geometry.GetOutput())
    trifilter.PassVertsOff()
    trifilter.PassLinesOff()
    trifilter.Update()

    return trifilter.GetOutput()


def unwrap(mesh, deep:bool=False):
    if isinstance(mesh, pyvista.PolyData):
        obj = vtkPolyData()
    elif isinstance(mesh, pyvista.UniformGrid):
        obj = vtkImageData()
    elif isinstance(mesh, pyvista.UnstructuredGrid):
        obj = vtkUnstructuredGrid()

    if deep:
        obj.DeepCopy(mesh)
    else:
        obj.ShallowCopy(mesh)
    return obj


def create_mesh_from_image_labels(image_labels: vtkImageData, label_num: int, output='uniform'):
    valid_output = ['uniform', 'voxel-like', 'smooth']
    mask = get_mask_from_labels(image_labels, label_num)
    if output == 'voxel-like':
        method = 'preserve_boundary'
        mesh = convert_voxels_to_poly(mask, method=method)
    elif output == 'smooth':
        method = 'flying_edges'
        mesh = convert_voxels_to_poly(mask, method=method)
        mesh = decimate_polydata(mesh)
        mesh = smooth_polydata(mesh, n_iterations=15)
    elif output == 'uniform':

        method = 'flying_edges'
        mesh = convert_voxels_to_poly(mask, method=method)
        mesh = decimate_polydata(mesh, target=60000)
        mesh = smooth_polydata(mesh, n_iterations=15)
        pvmesh = pyvista.wrap(mesh)
        import warnings  # disable warnings about using deprecated pyvista
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clus = pyacvd.Clustering(pvmesh)
            clus.subdivide(3)  # mesh is not dense enough for uniform remeshing
            clus.cluster(50000)
            pvremesh = clus.create_mesh()
        mesh = unwrap(pvremesh)
    else:
        return RuntimeError

    return mesh


def apply_mask_to_image(mask: vtkImageData, image: vtkImageData):
    image_scalars = get_scalars(image)
    mask_scalars = get_scalars(mask)
    assert(mask.GetScalarRange() == (0,1))
    applied = image_scalars * mask_scalars
    set_scalars(image, applied)
    return image


def get_neighbourhood(source: vtkPolyData, pt_id, breadth=1, max_dist=None):
    """
    Find the ids of the neighbours of pt_id.

    Control the size of the neighbourhood with the breadth parameter, e.g.:
        If breadth=1, return the point's immediate neighbours.
        If breadth=2, return the point's neighbours, as well as each neighbour's neighbours.

    Restrict the neighbourhood to be within a specified distance using max_dist.
    To select all points within max_dist radius of pt_id, breadth should be sufficiently high to ensure that a
    sufficient number of neighbours are initially found.
    max_dist and breadth correlate, and should both be increased/decreased proportionally for best results. The exact
    values to be chosen depends on the specific triangulation pattern and the average edge length. Longer edges mean
    that a smaller breadth is needed before sufficient points are found within max_dist.

    Args:
        pt_id: The point id
        source: Target mesh
        breadth: How "broad" the neighbourhood should be.
        max_dist: Neighbouring points further than this value will be excluded from the neighbourhood

    Returns:
        set of all neighbouring point IDs
    """

    n_hood = _get_neighbourhood(source, pt_id)
    for i in range(1, breadth):
        n_hood_copy = copy.deepcopy(n_hood)
        for point_id in n_hood_copy:
            local_n_hood = _get_neighbourhood(source, point_id)
            n_hood.update(local_n_hood)

    # filter out points which are too far
    if max_dist is not None:
        n_hood_copy = copy.deepcopy(n_hood)
        for neighbour_id in n_hood_copy:
            distance = compute_distance(source, pt_id, neighbour_id)
            if distance > max_dist:
                n_hood.remove(neighbour_id)

    return n_hood


def _get_neighbourhood(source: vtkPolyData, pt_id):
    """
    Find the ids of the neighbours of pt_id.

    :param pt_id: The point id.
    :return: The neighbour ids.
    """
    """
    Extract the topological neighbors for point pId. In two steps:
    1) source.GetPointCells(pt_id, cell_ids)
    2) source.GetCellPoints(cell_id, cell_point_ids) for all cell_id in cell_ids
    """
    cell_ids = vtkIdList()
    source.GetPointCells(pt_id, cell_ids)
    neighbour = set()
    for cell_idx in range(0, cell_ids.GetNumberOfIds()):
        cell_id = cell_ids.GetId(cell_idx)
        cell_point_ids = vtkIdList()
        source.GetCellPoints(cell_id, cell_point_ids)
        for cell_pt_idx in range(0, cell_point_ids.GetNumberOfIds()):
            neighbour.add(cell_point_ids.GetId(cell_pt_idx))
    return neighbour


def compute_distance(source: vtkPolyData, pt_id_a, pt_id_b):
    """
    Compute the distance between two points given their ids.

    :param pt_id_a:
    :param pt_id_b:
    :return:
    """
    pt_a = np.array(source.GetPoint(pt_id_a))
    pt_b = np.array(source.GetPoint(pt_id_b))
    return np.linalg.norm(pt_a - pt_b)


def get_scalars(source, point_id=None):
    if type(point_id) is int:
        return vtk_to_numpy(source.GetPointData().GetScalars())[point_id]
    elif hasattr(point_id, '__iter__'):
        return [vtk_to_numpy(source.GetPointData().GetScalars())[p] for p in point_id]
    elif point_id is None:
        return vtk_to_numpy(source.GetPointData().GetScalars())
    else:
        return RuntimeError


def compute_point_normals(source: vtkPolyData):
    normals = vtkPolyDataNormals()
    normals.SetInputData(source)
    normals.SetComputePointNormals(True)
    normals.SetComputeCellNormals(False)
    normals.SetSplitting(False)
    normals.Update()
    return normals.GetOutput()


def compute_cell_normals(source: vtkPolyData):
    normals = vtkPolyDataNormals()
    normals.SetInputData(source)
    normals.SetComputePointNormals(False)
    normals.SetComputeCellNormals(True)
    normals.SetSplitting(False)
    normals.Update()
    return normals.GetOutput()


def get_point_normals(source: vtkPolyData):
    if source.GetPointData().GetNormals() is None:
        source = compute_point_normals(source)
    return vtk_to_numpy(source.GetPointData().GetNormals())


def get_cell_normals(source: vtkPolyData):
    if source.GetCellData().GetNormals() is None:
        source = compute_cell_normals(source)
    return vtk_to_numpy(source.GetCellData().GetNormals())


def compute_furthest_distance_from_plane(plane: vtkPlane, points, constraint=None):
    valid_constraint = ['above', 'below', None]  # constrain to be above or below plane or either

    N = len(points)
    d = np.zeros(N)

    for i in range(N):
        d[i], _ = compute_signed_distance_to_plane(plane, points[i])

    if constraint is 'above':
        furthest_ind = np.argmax(d)
    elif constraint is 'below':
        furthest_ind = np.argmin(d)
    else:
        furthest_ind = np.argmax(np.abs(d))

    furthest_d = d[furthest_ind]
    furthest_point = points[furthest_ind]

    return furthest_d, np.asarray(furthest_point)


def minimize_local_scalar(source: vtkPolyData, initial_point, search_breadth, search_radius,
                          render=False,
                          render_camera=None,
                          render_save_path=None):
    """
    Minimize local point scalar values by evaluating median values within a small radius. Control the size of the local
    search area with the 'search_breadth' and 'search_radius' parameters, which control how a local neighbourhood of
    points is computed.
    Args:
        source:
        initial_point:
        search_breadth:
        search_radius:

    Returns:

    """
    # initialize
    _, _, min_point_id = find_closest_point(source, initial_point)
    min_n_hood = get_neighbourhood(source, min_point_id, breadth=search_breadth, max_dist=search_radius)
    min_scalars = get_scalars(source, min_n_hood)
    min_median = statistics.median(min_scalars)

    still_optimizing = True
    i = 0
    print("Minimizing surface curvature:")
    print('Iteration ' + str(i) + ': ' + str(min_median))
    while still_optimizing is True:
        i += 1
        initial_loop_median = min_median  # store for comparison later

        # search for lowest median scalars in local neighbourhood
        for point_id in min_n_hood:
            local_n_hood = get_neighbourhood(source, point_id, breadth=search_breadth, max_dist=search_radius)
            local_scalars = get_scalars(source, local_n_hood)
            local_median = statistics.median(local_scalars)
            if local_median < min_median:
                min_point_id = point_id
                min_median = local_median

        if min_median < initial_loop_median:
            min_n_hood = get_neighbourhood(source, min_point_id, breadth=search_breadth, max_dist=search_radius)
            still_optimizing = True
        else:
            still_optimizing = False

        print('Iteration ' + str(i) + ': ' + str(min_median))
    print('Done.')

    min_point = source.GetPoint(min_point_id)

    def render_minimization():

        renderer = vtkRenderer()
        if render_camera is None:
            camera = renderer.GetActiveCamera()
            camera.SetFocalPoint(min_point)
        else:
            renderer.SetActiveCamera(render_camera)

        plt_polydata(renderer, source, color='cornsilk')
        for p in min_n_hood:
            plt_point(renderer, source.GetPoint(p), radius=0.25, color='black')
        plt_point(renderer, initial_point, radius=1, color='red')
        plt_point(renderer, min_point, radius=1, color='green')

        show_scene(renderer, save_path=render_save_path)

    if render is True:
        render_minimization()

    return np.asarray(min_point)


def grow_mesh(source: vtkPolyData, seed_point_id, threshold=0.1, min_growth_rate=0.1):

    # initialize
    mesh_ids = set()
    mesh_ids.add(seed_point_id)

    iteration = 0
    stale_iteration = 0
    while True:
        candidate_points = get_set_neighbours(source, mesh_ids)
        # outliers = get_outliers(source, candidate_points, threshold)
        # if outliers is not None:
        #     modify_outlier_scalar(source, outliers)

        # grow mesh
        eligible_points = set()
        for p in candidate_points:
            scalar = get_scalars(source, p)
            if scalar < threshold:
                eligible_points.add(p)
            else:  # check if median value is within threshold
                n_hood = get_neighbourhood(source, p, breadth=2)
                n_hood.remove(p)  # strictly keep neighbours only
                scalars = get_scalars(source, n_hood)
                median = statistics.median(scalars)
                if median < threshold:
                    eligible_points.add(p)

        # ensure that eligible points have at least 2 neighbours in the original point_set (prevents run-away cases)
        if len(mesh_ids) > 1:
            min_num_neighbours = 2
            eligible_points_copy = copy.deepcopy(eligible_points)
            for p in eligible_points_copy:
                count = 0
                n_hood = get_neighbourhood(source, p)
                for pp in n_hood:
                    if pp in mesh_ids:
                        count += 1
                if count < min_num_neighbours and p in eligible_points:
                    eligible_points.remove(p)

        num_candidates = len(candidate_points)
        num_eligible = len(eligible_points)
        growth_rate = num_eligible / (num_candidates + 1e-8)

        print("Iteration " + str(iteration) + ": " + str(num_eligible) + "/" + str(num_candidates) + "=" + str(growth_rate))

        # create exit condition if growth regularly falls below 1.5 times min rate
        if growth_rate < min_growth_rate * 1.5:
            stale_iteration += 1
        else:
            stale_iteration = 0

        if growth_rate < min_growth_rate:
            break
        elif stale_iteration is 3:
            print("Growth rate is stale, exiting early.")
            break
        elif iteration > 50:
            print("WARNING: Mesh growth did not converge after 50 iterations.")
            break
        else:
            for p in eligible_points:
                mesh_ids.add(p)


        iteration += 1
    return mesh_ids


def grow_mesh_above_plane(source: vtkPolyData, initial_point_ids, plane, max_dist_from_origin):
    origin = np.asarray(plane.GetOrigin())
    final_point_ids = copy.deepcopy(initial_point_ids)
    num_added = 1  # arbitrarily initialize loop condition
    while num_added is not 0:
        num_added = 0
        set_neighbours = get_set_neighbours(source, final_point_ids)
        for p in set_neighbours:
            point = source.GetPoint(p)
            dist_to_plane, _ = compute_signed_distance_to_plane(plane, point)
            dist_to_origin = np.linalg.norm(point-origin, 2)

            is_above_plane = dist_to_plane > 0
            is_not_too_far = dist_to_origin < max_dist_from_origin
            if is_above_plane and is_not_too_far:  # add point
                num_added += 1
                final_point_ids.add(p)

    return final_point_ids


# def clip_mesh_with_plane(source: vtkPolyData, plane: vtkPlane()):
#     clipper = vtkClipPolyData()
#     clipper.SetInputData(source)
#     clipper.SetClipFunction(plane)
#     clipper.SetValue(0)
#     clipper.Update()
#     return clipper.GetOutput()


def compute_best_fit_plane(source: vtkPolyData, point_ids=None, point_away=True):
    if point_ids is None:
        N = source.GetNumberOfPoints()
        point_ids = np.arange(0, N)

    all_points = source.GetPoints()

    # create vtk id list from point ids
    ids = vtkIdList()
    for p in point_ids:
        ids.InsertNextId(p)

    # create vtk points from vtk id list
    plane_vtk_points = vtkPoints()
    all_points.GetPoints(ids, plane_vtk_points)

    origin = [0, 0, 0]
    normal = [0, 0, 1]
    plane = vtkPlane()
    plane.ComputeBestFittingPlane(plane_vtk_points, origin, normal)
    origin = np.asarray(origin)
    normal = np.asarray(normal)

    # make sure normal points away from surface
    if point_away is True:
        surface_point, distance, _ = find_closest_point(source, origin)
        surface_to_origin = origin-surface_point
        dot = np.dot(surface_to_origin, normal)
        if dot < 0:
            normal *= -1

    plane.SetNormal(normal[0], normal[1], normal[2])
    plane.SetOrigin(origin[0], origin[1], origin[2])

    return origin, normal, plane


def project_point_onto_plane(plane: vtkPlane, point):
    projected_point = [0, 0, 0]  # initialize
    plane.ProjectPoint(point, projected_point)
    return np.asarray(projected_point)


def compute_signed_distance_to_plane(plane: vtkPlane, query_points):
    if type(query_points) is tuple or len(query_points.shape) is 1:
        N = 1
        query_points = [query_points]

    else:
        N = len(query_points)

    projected_points = np.zeros((N, 3))
    distances = np.zeros(N)
    normal = np.asarray(plane.GetNormal())
    for i in range(N):
        distances[i] = plane.DistanceToPlane(query_points[i])
        projected_points[i] = project_point_onto_plane(plane, query_points[i])

        plane_to_point = query_points[i] - projected_points[i]
        dot = np.dot(plane_to_point, normal)

        if dot < 0:
            distances[i] *= -1

    if N is 1:
        projected_points = projected_points[0]
    return distances, projected_points

# def get_outliers(source: vtkPolyData, point_set, threshold):
#     """
#     Go through items in point_set and determine which points have scalar values > threshold (i.e. are outliers)
#     """
#     outliers = set()
#     for p in point_set:
#         scalar = get_scalars(source, p)
#         if scalar >= threshold:
#             outliers.add(p)
#
#     if len(outliers) is 0:
#         return None
#     else:
#         return outliers


def points_to_polydata(xyz):
    points = vtkPoints()
    # Create the topology of the point (a vertex)
    vertices = vtkCellArray()
    # Add points
    for i in range(len(xyz)):
        p = xyz[i]
        point_id = points.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)
    # Create a poly data object
    polydata = vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.Modified()

    return polydata


def convex_hull(apoly, alphaConstant=0):
    """
    Create a 3D Delaunay triangulation of input points.
    :param actor_or_list: can be either an ``Actor`` or a list of 3D points.
    :param float alphaConstant: For a non-zero alpha value, only verts, edges, faces,
        or tetra contained within the circumsphere (of radius alpha) will be output.
        Otherwise, only tetrahedra will be output.
    .. hint:: |convexHull| |convexHull.py|_
    """

    triangleFilter = vtkTriangleFilter()
    triangleFilter.SetInputData(apoly)
    triangleFilter.Update()
    poly = triangleFilter.GetOutput()

    delaunay = vtkDelaunay3D()  # Create the convex hull of the pointcloud
    if alphaConstant:
        delaunay.SetAlpha(alphaConstant)
    delaunay.SetInputData(poly)
    delaunay.SetTolerance(0.01)
    delaunay.Update()

    surfaceFilter = vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
    surfaceFilter.Update()

    return surfaceFilter.GetOutput()


def get_set_neighbours(source: vtkPolyData, point_set):
    """
    Extract a set of points which are neighbours to any of the points in point_set
    Args:
        source:
        point_set:

    Returns:

    """
    neighbours_set = copy.deepcopy(point_set)

    # dilate
    for p in point_set:
        n_hood = get_neighbourhood(source, p)
        for pp in n_hood:
            neighbours_set.add(pp)

    # remove original set points to extract new points only
    for p in point_set:
        neighbours_set.remove(p)

    return neighbours_set


# def modify_outlier_scalar(source: vtkPolyData, candidate_points):
#     new_values = []
#     for p in candidate_points:
#         # get median of neighbour scalars
#         n_hood = get_neighbourhood(source, p)
#         n_hood.remove(p)  # strictly keep neighbours only
#         scalars = get_scalars(source, n_hood)
#         median = statistics.median(scalars)
#
#         # we want to avoid modifying values before all median calcs are done because we don't want to use a modified
#         # value in the median of other points
#         # so, store in list for now
#         new_values.append(median)
#
#     for p_id, value in zip(candidate_points, new_values):
#         modify_scalar(source, p_id, value)


def set_scalars(source: vtkPolyData, new_value, point_ids=None):
    scalars_np = vtk_to_numpy(source.GetPointData().GetScalars())

    if point_ids is None:
        scalars_np = np.ones(scalars_np.shape) * new_value

    else:
        for p in point_ids:
            scalars_np[p] = new_value

    scalars_vtk = numpy_to_vtk(scalars_np)
    source.GetPointData().SetScalars(scalars_vtk)


def distance_to_polydata(polydata: vtkPolyData, point):
    # create tiny sphere at point location
    epsilon = np.finfo('float').eps
    sphere = vtkSphereSource()
    sphere.SetCenter(point)
    sphere.SetRadius(1e-4)
    sphere.SetPhiResolution(10)
    sphere.SetThetaResolution(10)
    sphere.Update()
    poly_point = sphere.GetOutput()

    distance_filter = vtkDistancePolyDataFilter()
    distance_filter.SetInputData(0, polydata)
    distance_filter.SetInputData(1, poly_point)
    distance_filter.Update()

    return distance_filter.GetOutput()


def compute_furthest_point_from_two_points(source: vtkPolyData, p1, p2):
    d1 = distance_to_polydata(source, p1)
    d2 = distance_to_polydata(source, p2)
    s1 = get_scalars(d1)
    s2 = get_scalars(d2)

    furthest_point_id = np.argmax(s1+s2)
    furthest_point = source.GetPoint(furthest_point_id)

    return np.asarray(furthest_point), furthest_point_id


# def compute_furthest_point_from_two_points_and_from_plane(source: vtkPolyData, plane, p1, p2):
#     points = get_points(source)
#     d_plane, _ = compute_signed_distance_to_plane(plane, points)
#     d_p1 = distance_to_polydata(source, p1)
#     d_p2 = distance_to_polydata(source, p2)
#     s1 = get_scalars(d_p1)
#     s2 = get_scalars(d_p2)
#
#     furthest_id1 = np.argmax(s1+s2+d_plane)
#     furthest_point1 = source.GetPoint(furthest_id1)
#
#     furthest_id2 = np.argmax(s1+s2+(d_plane*-1))
#     furthest_point2 = source.GetPoint(furthest_id2)
#
#     return np.asarray(furthest_point1), np.asarray(furthest_point2)