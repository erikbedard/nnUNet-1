

# Most of the vtk functions in this module are based  on code and examples from the following sources:

# Surface Extraction: Creating a mesh from pixel-data using Python and VTK
# https://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/

# Ray Casting with Python and VTK: Intersecting lines/rays with surface meshes
# https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/

# Mesh decimation example:
# https://kitware.github.io/vtk-examples/site/Python/Meshes/Decimation/

# Marching cubes + smoothing example:
# https://kitware.github.io/vtk-examples/site/Python/Visualization/FrogBrain/

from vtkmodules.all import *
from vtk.util.numpy_support import vtk_to_numpy

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2

import numpy as np


def _get_rgb(color):
    # see list of acceptable color names here:
    # https://htmlpreview.github.io/?https://github.com/Kitware/vtk-examples/blob/gh-pages/VTKNamedColorPatches.html
    if type(color) is str:
        colors = vtkNamedColors()
        return colors.GetColor3d(color)
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
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(_get_rgb(color))

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


# def get_scapula_markers(scapula, humerus):
#     p1, p2 = get_farthest_points(scapula)
#     acromion_angle, inferior_angle = identify_points(p1, p2, humerus)
#     return acromion_angle, inferior_angle


# def identify_points(p1, p2, humerus):
#     ''' Given two points (p1, p2) which define the diameter of the Scapula,
#     use the humerus to determine which of the two points is superior to the other'''
#     points = np.array(np.nonzero(humerus)).transpose()
#
#     # Find a convex hull in O(N log N)
#     hull = ConvexHull(points)
#
#     # Extract the points forming the hull
#     hull_points = points[hull.vertices, :]
#
#     d1 = get_shortest_distance(p1, hull_points)
#     d2 = get_shortest_distance(p2, hull_points)
#
#     # superior point is the one closest to humerus
#     if d1 < d2:
#         acromion_angle = p1
#         inferior_angle = p2
#     else:
#         acromion_angle = p2
#         inferior_angle = p1
#
#     return acromion_angle, inferior_angle


# def get_farthest_points(binary_mask):
#     """
#     Determine the two points in a binary mask which are the furthest apart
#     Args:
#         binary_mask:
#
#     Returns:
#         p1
#         p2
#
#     """
#     points = np.array(np.nonzero(binary_mask)).transpose()
#
#     # Find a convex hull in O(N log N)
#     hull = ConvexHull(points)
#
#     # Extract the points forming the hull
#     hull_points = points[hull.vertices, :]
#
#     # Naive way of finding the best pair in O(H^2) time if H is number of points on
#     # hull
#     hdist = cdist(hull_points, hull_points, metric='euclidean')
#
#     # Get the farthest apart points
#     best_pair = np.unravel_index(hdist.argmax(), hdist.shape)
#
#     p1 = hull_points[best_pair[0]].transpose()
#     p2 = hull_points[best_pair[1]].transpose()
#     return p1, p2


def convert_image_to_numpy(image: vtkImageData):
    rows, cols, _ = image.GetDimensions()
    values = image.GetPointData().GetScalars()
    np_image = vtk_to_numpy(values)
    np_image = np_image.reshape(rows, cols, -1)
    return np_image


def ray_casting(poly: vtkPolyData, point_source, point_target):
    # initialize oriented bounding box
    obbTree = vtkOBBTree()
    obbTree.SetDataSet(poly)
    obbTree.BuildLocator()

    # get intersection points
    pointsVTKintersection = vtkPoints()
    obbTree.IntersectWithLine(point_target,point_source, pointsVTKintersection, None)

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


def decimate_polydata(poly: vtkPolyData, reduction=0.9):
    """
    Down-sample (decimate) polydata with vtkDecimatePro.
    Args:
        poly:
        reduction:
    Returns:
        decimated vtkPolyData object
    """
    decimate = vtkDecimatePro()
    decimate.SetInputData(poly)
    decimate.SetTargetReduction(reduction)
    decimate.PreserveTopologyOn()
    decimate.Update()
    return decimate.GetOutput()


def smooth_polydata(poly: vtkPolyData, n_iterations=15, pass_band=0.001, feature_angle=120.0):
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
    valid_methods = ['flying_edges', 'marching_cubes', 'boundary']
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

    elif method is 'boundary':
        geometry = vtkImageDataGeometryFilter()
        geometry.SetInputData(binary_mask)
        geometry.Update()
        poly = geometry.GetOutput()
        ugrid = get_foreground_from_labels(poly, 1)

        surface = vtkGeometryFilter()
        surface.SetInputData(ugrid)

    surface.Update()
    return surface.GetOutput()


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