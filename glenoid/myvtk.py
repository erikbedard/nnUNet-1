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


def get_rgba(color, a=1):
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
        rgb = list(color)
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


def plt_polydata(renderer: vtkRenderer, polydata: vtkPolyData, color='tomato'):
    """
    Add polydata to a scene.
    Args:
        renderer:
        polydata:
        color:
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
    renderer.SetBackground(get_rgb(bg_color))

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


def create_mesh_from_image_labels(image_labels: vtkImageData, label_num: int, preserve_boundary=True):
    mask = get_mask_from_labels(image_labels, label_num)
    if preserve_boundary is True:
        method = 'preserve_boundary'
        mesh = convert_voxels_to_poly(mask, method=method)
    else:
        method = 'flying_edges'
        mesh = convert_voxels_to_poly(mask, method=method)
        mesh = decimate_polydata(mesh, target=10000)
        mesh = smooth_polydata(mesh, n_iterations=15)

    return mesh


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


def minimize_local_scalar(source: vtkPolyData, initial_point_id, search_breadth, search_radius):
    """
    Minimize local point scalar values by evaluating median values within a small radius. Control the size of the local
    search area with the 'search_breadth' and 'search_radius' parameters, which control how a local neighbourhood of
    points is computed.
    Args:
        source:
        initial_point_id:
        search_breadth:
        search_radius:

    Returns:

    """
    # initialize
    min_point_id = initial_point_id
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
    return min_point_id


def grow_mesh(source: vtkPolyData, seed_point_id, threshold=0.1, min_growth_rate=0.1):

    # initialize
    mesh_ids = set()
    mesh_ids.add(seed_point_id)

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
        print(num_candidates)
        print(num_eligible)
        growth_rate = num_eligible / num_candidates
        print(growth_rate)
        if growth_rate < min_growth_rate:
            break
        else:
            for p in eligible_points:
                mesh_ids.add(p)

    return mesh_ids


def compute_best_fit_plane(source: vtkPolyData, point_ids):
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
    surface_point, distance, _ = find_closest_point(source, origin)
    assert(distance > 0)
    surface_to_origin = origin-surface_point
    dot = np.dot(surface_to_origin,normal)
    if dot < 0:
        normal = normal * -1

    plane.SetNormal(normal[0], normal[1], normal[2])
    plane.SetOrigin(origin[0], origin[1], origin[2])

    return origin, normal, plane


def compute_signed_distance_to_plane(plane: vtkPlane, query_point):
    distance = plane.DistanceToPlane(query_point)

    projected_point = [0, 0, 0]
    plane.ProjectPoint(query_point, projected_point)

    plane_to_point = np.asarray(query_point) - np.asarray(projected_point)
    normal = np.asarray(plane.GetNormal())
    dot = np.dot(plane_to_point, normal)

    if dot > 0:
        return distance
    else:
        return distance * -1

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


def set_scalar(source: vtkPolyData, point_id, new_value):
    scalars_np = vtk_to_numpy(source.GetPointData().GetScalars())
    scalars_np[point_id] = new_value
    scalars_vtk = numpy_to_vtk(scalars_np)
    source.GetPointData().SetScalars(scalars_vtk)

