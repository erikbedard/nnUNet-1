import pyvista
from vtkmodules.all import *

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtk.util.numpy_support import *
import numpy as np


def _get_rgb(color):
    # input is either RGB triplet or named color
    # output is an RGB triplet
    #
    # see list of acceptable color names here:
    # https://htmlpreview.github.io/?https://github.com/Kitware/vtk-examples/blob/gh-pages/VTKNamedColorPatches.html
    if type(color) is str:
        colors = vtkNamedColors()
        return list(colors.GetColor3d(color))
    else:
        return color


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
    actor.GetProperty().SetColor(_get_rgb(color))

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
    actor.GetProperty().SetColor(_get_rgb(color))

    renderer.AddActor(actor)


def plt_polydata(renderer: vtkRenderer, polydata: vtkPolyData, color='tomato'):
    """
    Add polydata to a scene.
    Args:
        renderer:
        polydata:
        color:
    """
    rgb = _get_rgb(color)
    rgba = rgb.copy()
    rgba.append(1)
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


    renderer.AddActor(actor)


def show_scene(renderer: vtkRenderer, bg_color='white', window_name='VTK', window_size=(500, 500)):
    """
    Create and render a scene.
    Args:
        renderer:
        bg_color:
        window_name:
        window_size:
    """
    renderer.SetBackground(_get_rgb(bg_color))

    renWin = vtkRenderWindow()
    renWin.AddRenderer(renderer)
    renWin.SetSize(*window_size)
    renWin.SetWindowName(window_name)
    renWin.Render()

    iren = vtkRenderWindowInteractor()
    style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()


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


def save_stl(save_path, poly: vtkPolyData):
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


def decimate_polydata(poly: vtkPolyData, target=0.9):
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
    decimate.SetInputData(poly)
    decimate.SetTargetReduction(reduction)
    decimate.PreserveTopologyOn()
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

    return poly


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
    points = triangle_mesh.GetPoints()
    num_vertices = points.GetNumberOfPoints()
    vertex_list = []
    for i in range(0, num_vertices):
        point = points.GetPoint(i)
        vertex_list.append(point)

    return np.asarray(vertex_list), np.asarray(face_list)


def find_closest_point(point, poly: vtkPolyData):
    """
    Find point on polydata that is closest to the query point.
    Args:
        point: query point
        poly:

    Returns:
        closest point
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
    return closest_point, distance


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

    trifilter = vtkTriangleFilter()
    trifilter.SetInputData(geometry.GetOutput())
    trifilter.PassVertsOff()
    trifilter.PassLinesOff()
    trifilter.Update()

    return trifilter.GetOutput()
