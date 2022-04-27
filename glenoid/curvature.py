# this module borrows heavily from the vtk example:
# https://kitware.github.io/vtk-examples/site/Python/Visualization/CurvatureBandsWithGlyphs/

import numpy as np

from vtkmodules.all import *
from vtk.util import numpy_support
from vtkmodules.numpy_interface import dataset_adapter as dsa
import myvtk


def main():
    filepath = r"C:\erik\data\nnUNet\predictions\Task600\pred_imagesTr\final_output\ACD.nii.gz"
    labels = myvtk.read_nifti(filepath)
    scapula_label_num = 1
    mesh = myvtk.create_mesh_from_image_labels(labels, scapula_label_num, preserve_boundary=False)

    curvature_type = 'mean'
    curvature_str = "Mean_Curvature"
    curv_obj = myvtk.calc_curvature(mesh, curvature_type)

    # do_adjust_curve = True
    # if do_adjust_curve:
    #     adjust_edge_curvatures(curv_obj.GetOutput(), curvature_str)

    curv_obj.GetPointData().SetActiveScalars(curvature_str)
    scalar_range_curvatures = curv_obj.GetPointData().GetScalars(curvature_str).GetRange()

    lut = get_categorical_lut()
    lut.SetTableRange(scalar_range_curvatures)

    t = 0.08
    my_bands = [
        [scalar_range_curvatures[0], -t],
        [-t, t],
        [t, scalar_range_curvatures[1]]
    ]

    number_of_bands = lut.GetNumberOfTableValues()
    bands = get_custom_bands(scalar_range_curvatures, number_of_bands, my_bands)
    # Adjust the number of table values
    lut.SetNumberOfTableValues(len(bands))

    # Let's do a frequency table.
    # The number of scalars in each band.
    freq = get_frequencies(bands, curv_obj)
    bands, freq = adjust_ranges(bands, freq)
    print_bands_frequencies(bands, freq)

    lut.SetTableRange(scalar_range_curvatures)
    lut.SetNumberOfTableValues(len(my_bands))


    labels = [-t, 0, t]


    # Annotate
    values = vtkVariantArray()
    for i in range(len(labels)):
        values.InsertNextValue(vtkVariant(labels[i]))
    for i in range(values.GetNumberOfTuples()):
        lut.SetAnnotation(i, values.GetValue(i).ToString())

    # Create a lookup table with the colors reversed.
    lutr = reverse_lut(lut)

    # Create the contour bands.
    bcf = vtkBandedPolyDataContourFilter()
    bcf.SetInputData(curv_obj)
    # Use either the minimum or maximum value for each band.
    for k in bands:
        bcf.SetValue(k, bands[k][2])
    # We will use an indexed lookup table.
    bcf.SetScalarModeToIndex()
    bcf.GenerateContourEdgesOn()



    p = [179.0428772, 105.8322525, 133.18827057]
    seed_point, _, seed_point_id = myvtk.find_closest_point(curv_obj, p)

    s = myvtk.get_scalars(curv_obj, seed_point_id)

    breadth = 12
    max_dist = 10
    min_point_id = myvtk.minimize_local_scalar(curv_obj, seed_point_id, breadth, max_dist)
    min_point = curv_obj.GetPoint(min_point_id)

    n_hood = myvtk.get_neighbourhood(curv_obj, min_point_id, breadth=8, max_dist=5)




    x=1



    # mesh = bcf.GetOutput()
    #
    # from vtkmodules.vtkFiltersCore import vtkThreshold
    #
    # extract negative curves
    #
    # def threshold_mesh(mesh, lower_threshold, upper_threshold):
    #     selector = vtkThreshold()
    #     selector.SetInputArrayToProcess(0, 0, 0, vtkDataObject().FIELD_ASSOCIATION_CELLS,
    #                                     vtkDataSetAttributes().SCALARS)
    #     selector.SetInputData(mesh)
    #     selector.SetLowerThreshold(lower_threshold)
    #     selector.SetUpperThreshold(upper_threshold)
    #     selector.Update()
    #     return selector.GetOutput()
    #
    # lower_band_num = 1
    # upper_band_num = 3
    # curve_mesh1 = threshold_mesh(mesh, lower_band_num, lower_band_num)
    # curve_mesh2 = threshold_mesh(mesh, upper_band_num, upper_band_num)
    # cells = curved_mesh1.GetCellData()
    # scalars = numpy_support.vtk_to_numpy(cells.GetScalars())


    # ------------------------------------------------------------
    # Create the mappers and actors
    # ------------------------------------------------------------

    colors = vtkNamedColors()

    # Set the background color.
    colors.SetColor('BkgColor', colors.GetColor3d('White'))
    colors.SetColor("ParaViewBkg", [255, 255, 255, 255])

    src_mapper = vtkPolyDataMapper()
    src_mapper.SetInputConnection(bcf.GetOutputPort())
    src_mapper.SetScalarRange(scalar_range_curvatures)
    src_mapper.SetLookupTable(lut)
    src_mapper.SetScalarModeToUseCellData()

    src_actor = vtkActor()
    src_actor.SetMapper(src_mapper)

    # Create contour edges
    edge_mapper = vtkPolyDataMapper()
    edge_mapper.SetInputData(bcf.GetContourEdgesOutput())
    edge_mapper.SetResolveCoincidentTopologyToPolygonOffset()

    edge_actor = vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetColor(colors.GetColor3d('Black'))

    window_width = 800
    window_height = 800

    # Add scalar bars.
    scalar_bar = vtkScalarBarActor()
    # This LUT puts the lowest value at the top of the scalar bar.
    # scalar_bar->SetLookupTable(lut);
    # Use this LUT if you want the highest value at the top.
    scalar_bar.SetLookupTable(lutr)
    scalar_bar.SetTitle(curvature_str.replace('_', '\n'))
    scalar_bar.GetTitleTextProperty().SetColor(
        colors.GetColor3d('AliceBlue'))
    scalar_bar.GetLabelTextProperty().SetColor(
        colors.GetColor3d('AliceBlue'))
    scalar_bar.GetAnnotationTextProperty().SetColor(
        colors.GetColor3d('AliceBlue'))
    scalar_bar.UnconstrainedFontSizeOn()
    scalar_bar.SetMaximumWidthInPixels(window_width // 8)
    scalar_bar.SetMaximumHeightInPixels(window_height // 3)
    scalar_bar.SetPosition(0.85, 0.05)


    # ------------------------------------------------------------
    # Create the RenderWindow, Renderer and Interactor
    # ------------------------------------------------------------
    ren = vtkRenderer()
    ren_win = vtkRenderWindow()
    iren = vtkRenderWindowInteractor()
    style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    myvtk.plt_point(ren, seed_point, radius=0.5, color='blue')
    myvtk.plt_point(ren, min_point, radius=0.5, color='red')
    for p in n_hood:
        myvtk.plt_point(ren, curv_obj.GetPoint(p), radius=0.25, color='black')

    ren_win.AddRenderer(ren)
    # Important: The interactor must be set prior to enabling the widget.
    iren.SetRenderWindow(ren_win)

    # add actors
    ren.AddViewProp(src_actor)
    ren.AddViewProp(edge_actor)
    ren.AddActor2D(scalar_bar)

    ren.SetBackground(colors.GetColor3d('ParaViewBkg'))
    ren_win.SetSize(window_width, window_height)
    ren_win.SetWindowName('Scapula')

    camera = ren.GetActiveCamera()
    #camera.SetPosition(42.301174, 939.893457, -124.005030)
    camera.SetFocalPoint(min_point)
    #camera.SetViewUp(0.262286, -0.281321, -0.923073)
    #camera.SetDistance(789.297581)
    #camera.SetClippingRange(168.744328, 1509.660206)


    iren.Start()


def adjust_edge_curvatures(source, curvature_name, epsilon=1.0e-08):
    """
    This function adjusts curvatures along the edges of the surface by replacing
     the value with the average value of the curvatures of points in the neighborhood.

    Remember to update the vtkCurvatures object before calling this.

    :param source: A vtkPolyData object corresponding to the vtkCurvatures object.
    :param curvature_name: The name of the curvature, 'Gauss_Curvature' or 'Mean_Curvature'.
    :param epsilon: Absolute curvature values less than this will be set to zero.
    :return:
    """



    def compute_distance(pt_id_a, pt_id_b):
        """
        Compute the distance between two points given their ids.

        :param pt_id_a:
        :param pt_id_b:
        :return:
        """
        pt_a = np.array(source.GetPoint(pt_id_a))
        pt_b = np.array(source.GetPoint(pt_id_b))
        return np.linalg.norm(pt_a - pt_b)

    # Get the active scalars
    source.GetPointData().SetActiveScalars(curvature_name)
    np_source = dsa.WrapDataObject(source)
    curvatures = np_source.PointData[curvature_name]

    #  Get the boundary point IDs.
    array_name = 'ids'
    id_filter = vtkIdFilter()
    id_filter.SetInputData(source)
    id_filter.SetPointIds(True)
    id_filter.SetCellIds(False)
    id_filter.SetPointIdsArrayName(array_name)
    id_filter.SetCellIdsArrayName(array_name)
    id_filter.Update()

    edges = vtkFeatureEdges()
    edges.SetInputConnection(id_filter.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.Update()

    edge_array = edges.GetOutput().GetPointData().GetArray(array_name)
    boundary_ids = []
    for i in range(edges.GetOutput().GetNumberOfPoints()):
        boundary_ids.append(edge_array.GetValue(i))
    # Remove duplicate Ids.
    p_ids_set = set(boundary_ids)

    # Iterate over the edge points and compute the curvature as the weighted
    # average of the neighbours.
    count_invalid = 0
    for p_id in boundary_ids:
        p_ids_neighbors = myvtk.get_point_neighbourhood(p_id, source)
        # Keep only interior points.
        p_ids_neighbors -= p_ids_set
        # Compute distances and extract curvature values.
        curvs = [curvatures[p_id_n] for p_id_n in p_ids_neighbors]
        dists = [compute_distance(p_id_n, p_id) for p_id_n in p_ids_neighbors]
        curvs = np.array(curvs)
        dists = np.array(dists)
        curvs = curvs[dists > 0]
        dists = dists[dists > 0]
        if len(curvs) > 0:
            weights = 1 / np.array(dists)
            weights /= weights.sum()
            new_curv = np.dot(curvs, weights)
        else:
            # Corner case.
            count_invalid += 1
            # Assuming the curvature of the point is planar.
            new_curv = 0.0
        # Set the new curvature value.
        curvatures[p_id] = new_curv

    #  Set small values to zero.
    if epsilon != 0.0:
        curvatures = np.where(abs(curvatures) < epsilon, 0, curvatures)
        # Curvatures is now an ndarray
        curv = numpy_support.numpy_to_vtk(num_array=curvatures.ravel(),
                                          deep=True,
                                          array_type=VTK_DOUBLE)
        curv.SetName(curvature_name)
        source.GetPointData().RemoveArray(curvature_name)
        source.GetPointData().AddArray(curv)
        source.GetPointData().SetActiveScalars(curvature_name)


def constrain_curvatures(source, curvature_name, lower_bound=0.0, upper_bound=0.0):
    """
    This function constrains curvatures to the range [lower_bound ... upper_bound].

    Remember to update the vtkCurvatures object before calling this.

    :param source: A vtkPolyData object corresponding to the vtkCurvatures object.
    :param curvature_name: The name of the curvature, 'Gauss_Curvature' or 'Mean_Curvature'.
    :param lower_bound: The lower bound.
    :param upper_bound: The upper bound.
    :return:
    """

    bounds = list()
    if lower_bound < upper_bound:
        bounds.append(lower_bound)
        bounds.append(upper_bound)
    else:
        bounds.append(upper_bound)
        bounds.append(lower_bound)

    # Get the active scalars
    source.GetPointData().SetActiveScalars(curvature_name)
    np_source = dsa.WrapDataObject(source)
    curvatures = np_source.PointData[curvature_name]

    # Set upper and lower bounds.
    curvatures = np.where(curvatures < bounds[0], bounds[0], curvatures)
    curvatures = np.where(curvatures > bounds[1], bounds[1], curvatures)
    # Curvatures is now an ndarray
    curv = numpy_support.numpy_to_vtk(num_array=curvatures.ravel(),
                                      deep=True,
                                      array_type=VTK_DOUBLE)
    curv.SetName(curvature_name)
    source.GetPointData().RemoveArray(curvature_name)
    source.GetPointData().AddArray(curv)
    source.GetPointData().SetActiveScalars(curvature_name)


def get_color_series():
    color_series = vtkColorSeries()
    # Select a color scheme.
    # color_series_enum = color_series.BREWER_DIVERGING_BROWN_BLUE_GREEN_9
    # color_series_enum = color_series.BREWER_DIVERGING_SPECTRAL_10
    # color_series_enum = color_series.BREWER_DIVERGING_SPECTRAL_3
    # color_series_enum = color_series.BREWER_DIVERGING_PURPLE_ORANGE_9
    # color_series_enum = color_series.BREWER_SEQUENTIAL_BLUE_PURPLE_9
    # color_series_enum = color_series.BREWER_SEQUENTIAL_BLUE_GREEN_9
    color_series_enum = color_series.BREWER_QUALITATIVE_SET3
    # color_series_enum = color_series.CITRUS
    color_series.SetColorScheme(color_series_enum)
    return color_series


def get_categorical_lut():
    """
    Make a lookup table using vtkColorSeries.
    :return: An indexed (categorical) lookup table.
    """
    color_series = get_color_series()
    # Make the lookup table.
    lut = vtkLookupTable()
    color_series.BuildLookupTable(lut, color_series.CATEGORICAL)
    lut.SetNanColor(0, 0, 0, 1)
    return lut


def get_diverging_lut():
    """
    See: [Diverging Color Maps for Scientific Visualization](https://www.kennethmoreland.com/color-maps/)
                       start point         midPoint            end point
     cool to warm:     0.230, 0.299, 0.754 0.865, 0.865, 0.865 0.706, 0.016, 0.150
     purple to orange: 0.436, 0.308, 0.631 0.865, 0.865, 0.865 0.759, 0.334, 0.046
     green to purple:  0.085, 0.532, 0.201 0.865, 0.865, 0.865 0.436, 0.308, 0.631
     blue to brown:    0.217, 0.525, 0.910 0.865, 0.865, 0.865 0.677, 0.492, 0.093
     green to red:     0.085, 0.532, 0.201 0.865, 0.865, 0.865 0.758, 0.214, 0.233

    :return:
    """
    ctf = vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    # Cool to warm.
    ctf.AddRGBPoint(0.0, 0.085, 0.532, 0.201)
    ctf.AddRGBPoint(0.5, 0.865, 0.865, 0.865)
    ctf.AddRGBPoint(1.0, 0.758, 0.214, 0.233)

    table_size = 256
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(table_size)
    lut.Build()

    for i in range(0, table_size):
        rgba = list(ctf.GetColor(float(i) / table_size))
        rgba.append(1)
        lut.SetTableValue(i, rgba)

    return lut


def reverse_lut(lut):
    """
    Create a lookup table with the colors reversed.
    :param: lut - An indexed lookup table.
    :return: The reversed indexed lookup table.
    """
    lutr = vtkLookupTable()
    lutr.DeepCopy(lut)
    t = lut.GetNumberOfTableValues() - 1
    rev_range = reversed(list(range(t + 1)))
    for i in rev_range:
        rgba = [0.0] * 3
        v = float(i)
        lut.GetColor(v, rgba)
        rgba.append(lut.GetOpacity(v))
        lutr.SetTableValue(t - i, rgba)
    t = lut.GetNumberOfAnnotatedValues() - 1
    rev_range = reversed(list(range(t + 1)))
    for i in rev_range:
        lutr.SetAnnotation(t - i, lut.GetAnnotation(i))
    return lutr


def get_custom_bands(d_r, number_of_bands, my_bands):
    """
    Divide a range into custom bands.

    You need to specify each band as an list [r1, r2] where r1 < r2 and
    append these to a list.
    The list should ultimately look
    like this: [[r1, r2], [r2, r3], [r3, r4]...]

    :param: d_r - [min, max] the range that is to be covered by the bands.
    :param: number_of_bands - the number of bands, a positive integer.
    :return: A dictionary consisting of band number and [min, midpoint, max] for each band.
    """
    bands = dict()
    if (d_r[1] < d_r[0]) or (number_of_bands <= 0):
        return bands
    x = my_bands
    # Determine the index of the range minimum and range maximum.
    idx_min = 0
    for idx in range(0, len(my_bands)):
        if my_bands[idx][1] > d_r[0] >= my_bands[idx][0]:
            idx_min = idx
            break

    idx_max = len(my_bands) - 1
    for idx in range(len(my_bands) - 1, -1, -1):
        if my_bands[idx][1] > d_r[1] >= my_bands[idx][0]:
            idx_max = idx
            break

    # Set the minimum to match the range minimum.
    x[idx_min][0] = d_r[0]
    x[idx_max][1] = d_r[1]
    x = x[idx_min: idx_max + 1]
    for idx, e in enumerate(x):
        bands[idx] = [e[0], e[0] + (e[1] - e[0]) / 2, e[1]]
    return bands


def get_frequencies(bands, src):
    """
    Count the number of scalars in each band.
    The scalars used are the active scalars in the polydata.

    :param: bands - The bands.
    :param: src - The vtkPolyData source.
    :return: The frequencies of the scalars in each band.
    """
    freq = dict()
    for i in range(len(bands)):
        freq[i] = 0
    tuples = src.GetPointData().GetScalars().GetNumberOfTuples()
    for i in range(tuples):
        x = src.GetPointData().GetScalars().GetTuple1(i)
        for j in range(len(bands)):
            if x <= bands[j][2]:
                freq[j] += 1
                break
    return freq


def adjust_ranges(bands, freq):
    """
    The bands and frequencies are adjusted so that the first and last
     frequencies in the range are non-zero.
    :param bands: The bands dictionary.
    :param freq: The frequency dictionary.
    :return: Adjusted bands and frequencies.
    """
    # Get the indices of the first and last non-zero elements.
    first = 0
    for k, v in freq.items():
        if v != 0:
            first = k
            break
    rev_keys = list(freq.keys())[::-1]
    last = rev_keys[0]
    for idx in list(freq.keys())[::-1]:
        if freq[idx] != 0:
            last = idx
            break
    # Now adjust the ranges.
    min_key = min(freq.keys())
    max_key = max(freq.keys())
    for idx in range(min_key, first):
        freq.pop(idx)
        bands.pop(idx)
    for idx in range(last + 1, max_key + 1):
        freq.popitem()
        bands.popitem()
    old_keys = freq.keys()
    adj_freq = dict()
    adj_bands = dict()

    for idx, k in enumerate(old_keys):
        adj_freq[idx] = freq[k]
        adj_bands[idx] = bands[k]

    return adj_bands, adj_freq


def print_bands_frequencies(bands, freq, precision=2):
    prec = abs(precision)
    if prec > 14:
        prec = 14

    if len(bands) != len(freq):
        print('Bands and Frequencies must be the same size.')
        return
    s = f'Bands & Frequencies:\n'
    total = 0
    width = prec + 6
    for k, v in bands.items():
        total += freq[k]
        for j, q in enumerate(v):
            if j == 0:
                s += f'{k:4d} ['
            if j == len(v) - 1:
                s += f'{q:{width}.{prec}f}]: {freq[k]:8d}\n'
            else:
                s += f'{q:{width}.{prec}f}, '
    width = 3 * width + 13
    s += f'{"Total":{width}s}{total:8d}\n'
    print(s)


if __name__ == '__main__':

    main()
