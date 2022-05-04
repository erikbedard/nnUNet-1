import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import myvtk

from vtkmodules.vtkRenderingCore import vtkRenderer
import json
import glob
import os

def main():
    dataset_dir = r"C:\erik\data\prepared_datasets\Task600_TotalShoulder"
    labels_dir = os.path.join(dataset_dir, "labelsTs")

    root_save_dir = os.path.join(dataset_dir, "derivatives", "labelsTs")
    glenoid_labels_save_dir = os.path.join(root_save_dir, "glenoid_labelsTs")
    glenoid_labels_box_save_dir = os.path.join(root_save_dir, "glenoid_labelsTs_box")
    mesh_save_dir = os.path.join(root_save_dir, "meshes")

    os.makedirs(glenoid_labels_save_dir, exist_ok=True)
    os.makedirs(glenoid_labels_box_save_dir, exist_ok=True)
    os.makedirs(mesh_save_dir, exist_ok=True)

    # get lists of exiting files
    labels_paths = glob.glob(os.path.join(labels_dir, '*.nii.gz'))
    labels_paths.sort()

    # import multiprocessing
    # p = multiprocessing.Pool()
    # N = len(labels_paths)
    # data = zip(labels_paths, [glenoid_labels_save_dir]*N, [glenoid_labels_box_save_dir]*N, [mesh_save_dir]*N)
    # p.map(process_file, data)
    # p.close()
    for filepath in labels_paths:
        process_file(filepath, glenoid_labels_save_dir, glenoid_labels_box_save_dir, mesh_save_dir)


# def process_file(data):
#     filepath=data[0]
#     glenoid_labels_save_dir=data[1]
#     glenoid_labels_box_save_dir=data[2]
#     mesh_save_dir=data[3]
def process_file(filepath, glenoid_labels_save_dir, glenoid_labels_box_save_dir, mesh_save_dir):

    filename = os.path.basename(filepath).split('.')[0]
    print()
    print("Processing: " + filename)

    # get scapula from labels
    labels = myvtk.read_nifti(filepath)
    scapula_label_num = 1
    scapula_mask = myvtk.get_mask_from_labels(labels, scapula_label_num)
    scapula_mesh = myvtk.create_mesh_from_image_labels(labels, scapula_label_num, preserve_boundary=False)

    render_geometry = True
    render_glenoid_mesh = True
    render_segment = True

    # parameters for finding approximate glenoid centre
    optim_breadth = 12
    optim_max_dist = 10

    # parameters for mesh-growing algorithm
    growth_threshold = 0.1
    min_growth_rate = 0.1

    # parameters for performing  final segmentation geometry
    glenoid_thickness = 10
    segment_radius_scale = 1.05

    plane_origin, plane_normal, glenoid_mesh_ids = determine_glenoid_segment_parameters(
        scapula_mesh,
        optim_breadth=optim_breadth,
        optim_max_dist=optim_max_dist,
        growth_threshold=growth_threshold,
        min_growth_rate=min_growth_rate,
        glenoid_thickness=glenoid_thickness,
        segment_radius_scale=segment_radius_scale,
        render_geometry=render_geometry,
        render_glenoid_mesh=render_glenoid_mesh)

    # print("Segmenting glenoid...")
    glenoid_mask_box = myvtk.get_mask_from_labels(labels, scapula_label_num)  # start box as scapula
    glenoid_mask_segmented, glenoid_mask_box = extract_glenoid(glenoid_mask_box, scapula_mask, scapula_mesh, plane_origin, plane_normal, glenoid_mesh_ids, render=render_segment)
    #
    # print("Saving...")
    # glenoid_save_path = glenoid_labels_save_dir + os.path.sep + filename + ".nii.gz"
    # myvtk.save_nifti(glenoid_mask_segmented, glenoid_save_path)
    #
    # glenoid_save_path = glenoid_labels_box_save_dir + os.path.sep + filename + ".nii.gz"
    # myvtk.save_nifti(glenoid_mask_box, glenoid_save_path)


    # # #  finished creating glenoid segmentation label  # # #

    # save scapula meshes
    # mesh_voxels = myvtk.create_mesh_from_image_labels(labels, scapula_label_num, preserve_boundary=True)
    # save_path_voxels = mesh_save_dir + os.path.sep + filename + "_scapula_voxels.stl"
    # myvtk.save_stl(mesh_voxels, save_path_voxels)
    #
    # save_path_smooth = mesh_save_dir + os.path.sep + filename + "_scapula_smooth.stl"
    # myvtk.save_stl(scapula_mesh, save_path_smooth)
    #
    # # save glenoid meshes
    # glenoid_label = myvtk.read_nifti(glenoid_save_path)
    # glenoid_label_num = 1
    # #
    # mesh_voxels = myvtk.create_mesh_from_image_labels(glenoid_label, glenoid_label_num, preserve_boundary=True)
    # save_path_voxels = mesh_save_dir + os.path.sep + filename + "_glenoid_voxels.stl"
    # myvtk.save_stl(mesh_voxels, save_path_voxels)
    #
    # mesh_smooth = myvtk.create_mesh_from_image_labels(glenoid_label, glenoid_label_num, preserve_boundary=False)
    # save_path_smooth = mesh_save_dir + os.path.sep + filename + "_glenoid_smooth.stl"
    # myvtk.save_stl(mesh_smooth, save_path_smooth)
    #
    # # save humerus meshes
    # humerus_label_num = 2
    #
    # mesh_voxels = myvtk.create_mesh_from_image_labels(labels, humerus_label_num, preserve_boundary=True)
    # save_path_voxels = mesh_save_dir + os.path.sep + filename + "_humerus_voxels.stl"
    # myvtk.save_stl(mesh_voxels, save_path_voxels)
    #
    # mesh_smooth = myvtk.create_mesh_from_image_labels(labels, humerus_label_num, preserve_boundary=False)
    # save_path_smooth = mesh_save_dir + os.path.sep + filename + "_humerus_smooth.stl"
    # myvtk.save_stl(mesh_smooth, save_path_smooth)


def determine_glenoid_segment_parameters(scapula_mesh,
                                         optim_breadth=12,
                                         optim_max_dist=10,
                                         growth_threshold=0.1,
                                         min_growth_rate=0.1,
                                         glenoid_thickness=10,
                                         segment_radius_scale=1.2,
                                         render_geometry=False,
                                         render_glenoid_mesh=False):

    point_on_glenoid = get_initial_glenoid_point(scapula_mesh, render=render_geometry)

    glenoid_plane, glenoid_mesh_ids = compute_glenoid_plane(
        scapula_mesh,
        point_on_glenoid,
        optim_breadth=optim_breadth,
        optim_max_dist=optim_max_dist,
        growth_threshold=growth_threshold,
        min_growth_rate=min_growth_rate,
        render=render_glenoid_mesh)

    origin = np.asarray(glenoid_plane.GetOrigin())
    normal = np.asarray(glenoid_plane.GetNormal())

    # offset origin
    segment_origin = origin - normal * glenoid_thickness
    segment_normal = normal

    return segment_origin, segment_normal, glenoid_mesh_ids


def get_initial_glenoid_point(scapula_mesh, initialize='plane', render=False):
    valid_methods = ['line', 'plane']
    # line -> use a line between two farthest points on scapula for ray casting
    # plane -> use line method, but then cast points onto best-fit plane to the whole scapula

    # goal is to find a seed point on the glenoid

    # find acromion angle (AA) and inferior angle (IA) points by getting furthest two points in the mesh
    scapula_vertices, _ = myvtk.extract_mesh_data(scapula_mesh)
    p1, p2 = find_farthest_points(scapula_vertices)
    scapula_diameter = np.linalg.norm(p2 - p1, 2)

    if initialize is 'plane':
        N = scapula_mesh.GetNumberOfPoints()
        ids = np.arange(0, N-1)
        _, _, scapula_plane = myvtk.compute_best_fit_plane(scapula_mesh, ids)

        distance1, p1 = myvtk.compute_signed_distance_to_plane(scapula_plane, p1)
        distance2, p2 = myvtk.compute_signed_distance_to_plane(scapula_plane, p2)

        print("Using scapular plane initialization...")
        print("Initial points were moved by: " + str(distance1) + ", " + str(distance2))


    # cast a line between both points through the model and find intersections
    intersect_pts = myvtk.ray_casting(scapula_mesh, p1, p2)

    # insert p1 and p2 as assumed intersections
    intersect_pts = list(intersect_pts)
    intersect_pts.insert(0, p2)
    intersect_pts.append(p1)
    intersect_pts = np.asarray(intersect_pts)

    # we don't know which point (AA or IA) is which, so first find midpoints between all intersection points
    midpoints = get_mid_points(intersect_pts)

    # calc all distances from midpoints to scapula surface
    num_midpoints = midpoints.shape[0]
    distances = np.zeros(num_midpoints)
    closest_points = np.zeros(midpoints.shape)
    for i in range(num_midpoints):
        query = midpoints[i]
        closest_points[i], distances[i], _ = myvtk.find_closest_point(scapula_mesh, query)

    # the midpoint (MP) which is furthest from the scapula must lie between glenoid and acromion
    midpoint_is_valid = False
    while not midpoint_is_valid:
        # we want the midpoint furthest away from scapula that is also within 1/4 distance of diameter
        ind = np.argmax(distances)
        midpoint_near_glenoid = midpoints[ind]
        d_from_p1 = np.linalg.norm(midpoint_near_glenoid - p1, 2)
        d_from_p2 = np.linalg.norm(midpoint_near_glenoid - p2, 2)
        if d_from_p1 < scapula_diameter / 4 or d_from_p2 < scapula_diameter / 4:
            midpoint_is_valid = True
        else:
            distances[ind] = -100


    # we can now identify the AA as the point that is closer to MP than IA and get a point on glenoid
    d1 = np.linalg.norm(p1 - midpoint_near_glenoid, 2)
    d2 = np.linalg.norm(p2 - midpoint_near_glenoid, 2)
    if d1 < d2:
        lateral_acromion = p1
        inferior_angle = p2
    else:
        lateral_acromion = p2
        inferior_angle = p1

    point_on_glenoid = myvtk.ray_casting(scapula_mesh, inferior_angle, midpoint_near_glenoid)[0]

    def render_geometry(scapula_mesh,inferior_angle,lateral_acromion,midpoint_near_glenoid):
        renderer = vtkRenderer()
        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(point_on_glenoid)

        myvtk.plt_polydata(renderer, scapula_mesh, color='cornsilk')
        myvtk.plt_point(renderer, inferior_angle, radius=1, color='black')
        myvtk.plt_point(renderer, lateral_acromion, radius=1, color='black')
        myvtk.plt_point(renderer, midpoint_near_glenoid, radius=1, color='black')
        myvtk.plt_point(renderer, point_on_glenoid, radius=1, color='blue')

        myvtk.plt_line(renderer, p1, p2, color='red')
        myvtk.show_scene(renderer)

    if render is True:
        render_geometry(scapula_mesh, inferior_angle, lateral_acromion, midpoint_near_glenoid)

    return point_on_glenoid


def compute_glenoid_plane(scapula_mesh,
                          initial_point,
                          optim_breadth=12,
                          optim_max_dist=10,
                          growth_threshold=0.1,
                          min_growth_rate=0.1,
                          render=False):
    # find approximate center of glenoid
    mesh_curves = myvtk.calc_curvature(scapula_mesh, method='mean')
    _, _, seed_point_id = myvtk.find_closest_point(scapula_mesh, initial_point)
    glenoid_seed_point_id = myvtk.minimize_local_scalar(mesh_curves, seed_point_id, optim_breadth, optim_max_dist)

    # create glenoid mesh
    glenoid_mesh_ids = myvtk.grow_mesh(mesh_curves, glenoid_seed_point_id,
                                       threshold=growth_threshold, min_growth_rate=min_growth_rate)

    # get plane
    origin, normal, glenoid_plane = myvtk.compute_best_fit_plane(scapula_mesh, glenoid_mesh_ids)

    # calc max distance from plane origin
    glenoid_points = []
    for p in glenoid_mesh_ids:
        glenoid_points.append(scapula_mesh.GetPoint(p))
    p1, p2 = find_farthest_points(np.asarray(glenoid_points))
    _, p1 = myvtk.compute_signed_distance_to_plane(glenoid_plane, p1)
    _, p2 = myvtk.compute_signed_distance_to_plane(glenoid_plane, p2)
    dist_to_p1 = np.linalg.norm(p1-origin, 2)
    dist_to_p2 = np.linalg.norm(p2-origin, 2)

    glenoid_diameter = np.linalg.norm(p1-p2, 2)

    if dist_to_p1 > dist_to_p2:
        max_dist_from_origin = dist_to_p1
    else:
        max_dist_from_origin = dist_to_p2

    # expand mesh based on plane and max-dist
    max_dist = max_dist_from_origin*1.1  # allow 10% tolerance
    glenoid_mesh_ids = myvtk.grow_mesh_above_plane(scapula_mesh, glenoid_mesh_ids, glenoid_plane, max_dist)

    # recompute plane from expanded mesh
    origin, normal, glenoid_plane = myvtk.compute_best_fit_plane(scapula_mesh, glenoid_mesh_ids)


    def render_mesh_growing(scapula_mesh, seed_point_id, mesh_ids, plane_origin, plane_normal):

        renderer = vtkRenderer()
        camera = renderer.GetActiveCamera()
        glenoid_origin, _, _ = myvtk.find_closest_point(scapula_mesh, plane_origin)
        camera.SetFocalPoint(glenoid_origin)

        myvtk.plt_polydata(renderer, scapula_mesh, color='cornsilk')
        for p in mesh_ids:
            myvtk.plt_point(renderer, scapula_mesh.GetPoint(p), radius=0.25, color='black')
        myvtk.plt_point(renderer, scapula_mesh.GetPoint(seed_point_id), radius=0.25, color='red')
        myvtk.plt_point(renderer, plane_origin, radius=0.25, color='green')
        myvtk.plt_line(renderer, plane_origin, plane_origin + (plane_normal * 5))
        myvtk.show_scene(renderer)

    if render is True:
        render_mesh_growing(scapula_mesh, seed_point_id, glenoid_mesh_ids, origin, normal)

    return glenoid_plane, glenoid_mesh_ids

from vtkmodules.all import *
from vtk.util.numpy_support import *
def extract_glenoid(glenoid_mask_box, scapula_mask, scapula_mesh, origin, normal, glenoid_mesh_ids, render=False):

    plane = myvtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    # create a 3D convex hull from glenoid surface points and their projections onto the segment plane
    glenoid_shape_points = []
    for p in glenoid_mesh_ids:
        point = scapula_mesh.GetPoint(p)
        _, projected_point = myvtk.compute_signed_distance_to_plane(plane, point)
        glenoid_shape_points.append(point)
        glenoid_shape_points.append(projected_point)

    glenoid_shape_points = np.asarray(glenoid_shape_points)
    poly = myvtk.points_to_polydata(glenoid_shape_points)
    hull = myvtk.convex_hull(poly)


    normals = vtkPolyDataNormals()
    normals.SetInputData(hull)
    normals.SetComputePointNormals(True)
    normals.Update()
    hull = normals.GetOutput()
    normals = vtk_to_numpy(hull.GetPointData().GetNormals())
    vtk_pts = hull.GetPoints()
    for i in range(hull.GetNumberOfPoints()):
        old_point = vtk_pts.GetPoint(i)
        point_normal = normals[i]
        new_point = old_point + 5 * point_normal

        # any new points below the plane should have direction changed along the plane
        # is_pointing_down = np.all(np.isclose(-point_normal, normal))
        # d_to_plane, projected_point = myvtk.compute_signed_distance_to_plane(plane, new_point)
        # if d_to_plane < 0.0001 and not is_pointing_down:
        #     point_normal = projected_point - old_point
        #     point_normal /= np.linalg.norm(point_normal, 2)

        # new_point = old_point + 5 * point_normal

        vtk_pts.SetPoint(i, new_point)

    hull2 = myvtk.convex_hull(hull)
    implicit_function = vtkImplicitPolyDataDistance()
    implicit_function.SetInput(hull2)

    dilate_erode = vtkImageDilateErode3D()
    dilate_erode.SetInputData(glenoid_mask_box)
    dilate_erode.SetDilateValue(1)
    dilate_erode.SetErodeValue(0)
    dilate_erode.SetKernelSize(20, 20, 20)
    dilate_erode.ReleaseDataFlagOff()
    dilate_erode.Update()

    glenoid_mask_box = dilate_erode.GetOutput()
    scalars = myvtk.get_scalars(glenoid_mask_box)
    scapula_ind = np.argwhere(scalars)
    for i in scapula_ind:
        point = np.asarray(glenoid_mask_box.GetPoint(i))
        if implicit_function.FunctionValue(point) <= 0:  # inside hull
            d_to_plane, _ = myvtk.compute_signed_distance_to_plane(plane, point)

            # must be on right side of plane
            if d_to_plane > 0:
                continue

        # point is in hull and above plane. Hooray! it's the glenoid, set it to foreground
        myvtk.set_scalar(glenoid_mask_box, i, 0)

    # outside hull



    hull2_vertices, _ = myvtk.extract_mesh_data(hull2)
    p1, p2 = find_farthest_points(hull2_vertices)
    glenoid_diameter = np.linalg.norm(p2 - p1, 2)

    # loop through all scapula pixels and determine which are part of glenoid
    scalars = myvtk.get_scalars(scapula_mask)
    scapula_ind = np.argwhere(scalars)
    background = 0
    for i in scapula_ind:
        point = np.asarray(scapula_mask.GetPoint(i))
        d_to_plane, projected_point = myvtk.compute_signed_distance_to_plane(plane, point)

        # must be on right side of plane
        wrong_side_of_plane = d_to_plane < 0

        # must be not be too far above plane
        max_dist_above_plane = 20
        too_far_from_plane = d_to_plane > max_dist_above_plane

        # must not be too far from hull
        d_to_origin = np.linalg.norm(origin - point, 2)
        too_far_from_origin = d_to_origin > glenoid_diameter / 1.9

        is_glenoid = False
        if wrong_side_of_plane or too_far_from_plane or too_far_from_origin:
            myvtk.set_scalar(scapula_mask, i, background)  # point is not glenoid
            continue
        else:
            # projected point must be inside the convex hull
            # max_dist_to_hull = 5
            # closest_point, d_to_hull, _ = myvtk.find_closest_point(hull, point)
            # too_far_from_hull = d_to_hull > max_dist_to_hull

            if implicit_function.FunctionValue(point) <= 0:
                # inside hull
                pass
            else:
                # outside hull
                myvtk.set_scalar(scapula_mask, i, background)  # point is not glenoid

    glenoid_mask_segmented = scapula_mask

    def render_glenoid(glenoid_mask_segmented, segment_origin):
        foreground = 1
        glenoid_mesh = myvtk.create_mesh_from_image_labels(glenoid_mask_segmented, foreground, preserve_boundary=True)

        renderer = vtkRenderer()
        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(segment_origin)
        myvtk.plt_polydata(renderer, glenoid_mesh, color='cornsilk')
        myvtk.show_scene(renderer)

    if render is True:
        render_glenoid(glenoid_mask_segmented, origin)

    return glenoid_mask_segmented, glenoid_mask_box


def find_farthest_points(points):
    # TODO: rewrite this using vtk functions
    """
    Determine the two points which are the furthest apart from each other.
    Args:
        points
    Returns:
        p1
        p2
    """
    # Find a convex hull in O(N log N)
    hull = ConvexHull(points)

    # Extract the points forming the hull
    hull_points = points[hull.vertices, :]

    # Naive way of finding the best pair in O(H^2) time if H is number of points on
    # hull
    hdist = cdist(hull_points, hull_points, metric='euclidean')

    # Get the farthest apart points
    best_pair = np.unravel_index(hdist.argmax(), hdist.shape)

    p1 = hull_points[best_pair[0]].transpose()
    p2 = hull_points[best_pair[1]].transpose()
    return p1, p2


def get_mid_points(points):
    # takes list of N 3D tuples as input  and computes midpoints between each sequential pair of points
    # returns list of (N-1) tuples of mid points

    N = points.shape[0]
    D = points.shape[1]
    mid_points = np.zeros((N-1, D))
    for i in range(N-1):
        p1 = points[i, :]
        p2 = points[i+1, :]
        distance = np.linalg.norm(p1-p2, 2)
        direction = (p2-p1)/distance
        mid_points[i, :] = p1 + direction * distance/2
    return mid_points



if __name__ == '__main__':
    main()