import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import myvtk

from vtkmodules.vtkRenderingCore import vtkRenderer
import glob
import os

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2

def main():
    ts_or_tr = "labelsTs_ignored"
    dataset_dir = r"C:\erik\data\prepared_datasets\Task600_TotalShoulder"
    labels_dir = os.path.join(dataset_dir, ts_or_tr)

    root_save_dir = os.path.join(dataset_dir, "derivatives", ts_or_tr)
    glenoid_labels_save_dir = os.path.join(root_save_dir, "glenoid_labels")
    glenoid_mask_save_dir = os.path.join(root_save_dir, "glenoid_masks")
    mesh_save_dir = os.path.join(root_save_dir, "meshes")
    processing_save_dir = os.path.join(root_save_dir, "processing")

    os.makedirs(glenoid_labels_save_dir, exist_ok=True)
    os.makedirs(glenoid_mask_save_dir, exist_ok=True)
    os.makedirs(mesh_save_dir, exist_ok=True)
    os.makedirs(processing_save_dir, exist_ok=True)

    # get lists of exiting files
    labels_paths = glob.glob(os.path.join(labels_dir, '*.nii.gz'))
    labels_paths.sort()
    # labels_paths = [labels_paths[12]]

    import multiprocessing
    p = multiprocessing.Pool()
    N = len(labels_paths)
    data = zip(labels_paths,
               [glenoid_labels_save_dir]*N,
               [glenoid_mask_save_dir]*N,
               [mesh_save_dir]*N,
               [processing_save_dir]*N)
    p.map(process_file, data)
    p.close()
 #   for filepath in labels_paths:
 #       process_file(filepath, glenoid_labels_save_dir, glenoid_mask_save_dir, mesh_save_dir, processing_save_dir)


def process_file(data):
    filepath=data[0]
    glenoid_labels_save_dir=data[1]
    glenoid_mask_save_dir=data[2]
    mesh_save_dir=data[3]
    processing_save_dir=data[4]
#def process_file(filepath, glenoid_labels_save_dir, glenoid_mask_save_dir, mesh_save_dir, processing_save_dir):

    filename = os.path.basename(filepath).split('.')[0]
    print()
    print("Processing: " + filename)

    # get scapula from labels
    labels = myvtk.read_nifti(filepath)
    scapula_label_num = 1
    scapula_mask = myvtk.get_mask_from_labels(labels, scapula_label_num)
    scapula_mesh = myvtk.create_mesh_from_image_labels(labels, scapula_label_num, preserve_boundary=False)

    render_geometry = True
    render_minimization = True
    render_glenoid_mesh = True
    render_segment = True

    # find point on glenoid
    render_save_path = os.path.join(processing_save_dir, filename + "_1_geometry.png")
    point_on_glenoid, camera = get_initial_glenoid_point(
        scapula_mesh,
        render=render_geometry,
        render_save_path=render_save_path)

    # find approximate center of glenoid
    breadth = 8
    max_dist = 5  # mm
    mesh_curves = myvtk.calc_curvature(scapula_mesh, method='mean')

    render_save_path = os.path.join(processing_save_dir, filename + "_2_minimize.png")
    min_point_on_glenoid = myvtk.minimize_local_scalar(
        mesh_curves,
        point_on_glenoid,
        breadth,
        max_dist,
        render=render_minimization,
        render_camera=camera,
        render_save_path=render_save_path)

    threshold = 0.1
    min_growth_rate = 0.1

    render_save_path = os.path.join(processing_save_dir, filename + "_3_mesh.png")
    glenoid_plane, glenoid_mesh_ids = compute_glenoid_plane(
        mesh_curves,
        min_point_on_glenoid,
        threshold=threshold,
        min_growth_rate=min_growth_rate,
        render=render_glenoid_mesh,
        render_camera=camera,
        render_save_path=render_save_path)

    print("Segmenting glenoid...")
    glenoid_thickness = 10  # mm
    buffer_distance = 5  # mm

    # offset origin
    origin = np.asarray(glenoid_plane.GetOrigin())
    normal = np.asarray(glenoid_plane.GetNormal())
    segment_origin = origin - normal * glenoid_thickness

    glenoid_mask = myvtk.create_blank_image_from_source(scapula_mask)

    render_save_path = os.path.join(processing_save_dir, filename + "_4_mask.png")
    glenoid_mask = create_glenoid_mask(
        glenoid_mask,
        scapula_mesh,
        glenoid_mesh_ids,
        segment_origin,
        normal,
        buffer_distance=buffer_distance,
        render=render_segment,
        render_camera=camera,
        render_save_path=render_save_path)

    print("Saving...")
    # save glenoid mask image
    glenoid_save_path = os.path.join(glenoid_mask_save_dir, filename + ".nii.gz")
    myvtk.save_nifti(glenoid_mask, glenoid_save_path)

    # save segmented glenoid image
    glenoid_segmented = myvtk.apply_mask_to_image(glenoid_mask, scapula_mask)
    glenoid_save_path = os.path.join(glenoid_labels_save_dir, filename + ".nii.gz")
    myvtk.save_nifti(glenoid_segmented, glenoid_save_path)

    # save glenoid meshes
    foreground = 1
    mesh_voxels = myvtk.create_mesh_from_image_labels(glenoid_segmented, foreground, preserve_boundary=True)
    save_path_voxels = mesh_save_dir + os.path.sep + filename + "_glenoid.stl"
    myvtk.save_stl(mesh_voxels, save_path_voxels)

    mesh_smooth = myvtk.create_mesh_from_image_labels(glenoid_mask, foreground, preserve_boundary=True)
    save_path_smooth = mesh_save_dir + os.path.sep + filename + "_glenoid_mask.stl"
    myvtk.save_stl(mesh_smooth, save_path_smooth)


    # save scapula meshes
    # mesh_voxels = myvtk.create_mesh_from_image_labels(labels, scapula_label_num, preserve_boundary=True)
    # save_path_voxels = mesh_save_dir + os.path.sep + filename + "_scapula_voxels.stl"
    # myvtk.save_stl(mesh_voxels, save_path_voxels)
    #
    # save_path_smooth = mesh_save_dir + os.path.sep + filename + "_scapula_smooth.stl"
    # myvtk.save_stl(scapula_mesh, save_path_smooth)
    #
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


def get_initial_glenoid_point(scapula_mesh, initialize='scapular-plane', render=False, render_save_path=None):
    valid_methods = ['sup-inf', 'triple-point', 'ant-post', 'scapular-plane']
    # line -> use a line between two farthest points on scapula for ray casting
    # plane -> use line method, but then cast points onto best-fit plane to the whole scapula

    # goal is to find a seed point on the glenoid

    # find acromion angle (AA) and inferior angle (IA) points by getting furthest two points in the mesh
    scapula_points = myvtk.get_points(scapula_mesh)
    p1, p2 = find_farthest_points(scapula_points)
    scapula_diameter = np.linalg.norm(p2 - p1, 2)

    # get initial plane
    _, _, scapula_plane = myvtk.compute_best_fit_plane(scapula_mesh)

    # get furthest points from plane (+ve and -ve)
    scapula_points = myvtk.get_points(scapula_mesh)
    d_above , p3 = myvtk.compute_furthest_distance_from_plane(scapula_plane, scapula_points, constraint='above')
    d_below, p4 = myvtk.compute_furthest_distance_from_plane(scapula_plane, scapula_points, constraint='below')
    p3p4 = (p3+p4)/2

    # we can now identify the AA as the point that is closer to MP than IA and get a point on glenoid
    d1 = np.linalg.norm(p1 - p3p4, 2)
    d2 = np.linalg.norm(p2 - p3p4, 2)
    if d1 < d2:
        lateral_acromion = p1
        inferior_angle = p2
    else:
        lateral_acromion = p2
        inferior_angle = p1

    ant_post_point = (p3 + p4) / 2
    triple_point = (lateral_acromion + p3 + p4)/3

    medial_point, _ = myvtk.compute_furthest_point_from_two_points(scapula_mesh, lateral_acromion, inferior_angle)

    if initialize == 'scapular-plane':

        # # extract points within a percentile of distance from plane
        # N = scapula_mesh.GetNumberOfPoints()
        # distances = np.zeros(N)
        # for i in range(N):
        #     point = scapula_mesh.GetPoint(i)
        #     d, _ = myvtk.compute_signed_distance_to_plane(scapula_plane, point)
        #     distances[i] = abs(d)
        #
        # percentile = np.percentile(distances, 90)
        # close_points_ind = np.where(distances < percentile)[0]
        # points = myvtk.get_points(scapula_mesh, close_points_ind)
        #
        # poly = myvtk.points_to_polydata(points)
        # _, _, scapula_plane = myvtk.compute_best_fit_plane(poly,point_away=False)

        distance1, ray_point1 = myvtk.compute_signed_distance_to_plane(scapula_plane, lateral_acromion)
        distance2, ray_point2 = myvtk.compute_signed_distance_to_plane(scapula_plane, inferior_angle)

        print("Using scapular plane initialization...")
        print("Initial points were moved by: " + str(distance1) + ", " + str(distance2))
    elif initialize == 'triple-point':
        ray_point1 = triple_point
        ray_point2 = inferior_angle
    elif initialize == 'sup-inf':
        ray_point1 = lateral_acromion
        ray_point2 = inferior_angle
    elif initialize == 'ant-post':
        ray_point1 = ant_post_point
        ray_point2 = inferior_angle
    else:
        return RuntimeError

    # cast a line between both points through the model and find intersections
    intersect_pts = myvtk.ray_casting(scapula_mesh, ray_point1, ray_point2)

    # insert p1 and p2 as assumed intersections
    intersect_pts = list(intersect_pts)
    intersect_pts.insert(0, ray_point2)
    intersect_pts.append(ray_point1)
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
    midpoint_near_glenoid = None
    while not midpoint_is_valid:
        # we want the midpoint furthest away from scapula that is also within 1/4 distance of diameter
        ind = np.argmax(distances)
        midpoint_near_glenoid = midpoints[ind]
        d_from_p1 = np.linalg.norm(midpoint_near_glenoid - ray_point1, 2)
        d_from_p2 = np.linalg.norm(midpoint_near_glenoid - ray_point2, 2)
        if d_from_p1 < scapula_diameter / 4 or d_from_p2 < scapula_diameter / 4:
            midpoint_is_valid = True
        else:
            distances[ind] = -100

    point_on_glenoid = myvtk.ray_casting(scapula_mesh, inferior_angle, midpoint_near_glenoid)[0]

    def render_geometry():
        focal = point_on_glenoid
        direction = focal-(medial_point+inferior_angle)/2
        direction /= np.linalg.norm(direction)
        position = focal + 200*direction

        y_axis_point = (0,1,0)
        proj_pt = myvtk.project_point_onto_plane(scapula_plane, y_axis_point)

        renderer = vtkRenderer()
        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(focal)
        camera.SetPosition(position)
        # camera.ComputeViewPlaneNormal()
        camera.SetViewUp(lateral_acromion)

        myvtk.plt_polydata(renderer, scapula_mesh, color='cornsilk')
        myvtk.plt_plane(renderer, scapula_plane, ray_point1, ray_point2, color='PaleTurquoise')
        myvtk.plt_point(renderer, inferior_angle, radius=1, color='black')
        myvtk.plt_point(renderer, lateral_acromion, radius=1, color='black')
        myvtk.plt_point(renderer, p3, radius=1, color='black')
        myvtk.plt_point(renderer, p4, radius=1, color='black')
        myvtk.plt_point(renderer, medial_point, radius=1, color='black')

        #myvtk.plt_point(renderer, midpoint_near_glenoid, radius=1, color='black')

        myvtk.plt_point(renderer, ant_post_point, radius=1, color='green')
        myvtk.plt_line(renderer, ant_post_point, inferior_angle, color='green')

        myvtk.plt_point(renderer, triple_point, radius=1, color='red')
        myvtk.plt_line(renderer, triple_point, inferior_angle, color='red')

        myvtk.plt_point(renderer, lateral_acromion, radius=1, color='black')
        myvtk.plt_line(renderer, lateral_acromion, inferior_angle, color='black')

        myvtk.plt_point(renderer, point_on_glenoid, radius=1, color='blue')
        myvtk.plt_line(renderer, ray_point1, ray_point2, color='blue')

        myvtk.show_scene(renderer, save_path=render_save_path)

        return camera

    if render is True:
        camera = render_geometry()
    else:
        camera = None

    return point_on_glenoid, camera


def compute_glenoid_plane(scapula_curves,
                          initial_point,
                          threshold=0.1,
                          min_growth_rate=0.1,
                          render=False,
                          render_camera=None,
                          render_save_path=None):

    _, _, initial_point_id = myvtk.find_closest_point(scapula_curves, initial_point)

    # create glenoid mesh
    glenoid_mesh_ids = myvtk.grow_mesh(
        scapula_curves,
        initial_point_id,
        threshold=threshold,
        min_growth_rate=min_growth_rate)

    # get plane
    origin, normal, glenoid_plane = myvtk.compute_best_fit_plane(scapula_curves, glenoid_mesh_ids)

    # calc max distance from plane origin
    glenoid_points = []
    for p in glenoid_mesh_ids:
        glenoid_points.append(scapula_curves.GetPoint(p))
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
    glenoid_mesh_ids = myvtk.grow_mesh_above_plane(scapula_curves, glenoid_mesh_ids, glenoid_plane, max_dist)

    # recompute plane from expanded mesh
    origin, normal, glenoid_plane = myvtk.compute_best_fit_plane(scapula_curves, glenoid_mesh_ids)


    def render_mesh_growing():

        renderer = vtkRenderer()
        if render_camera is None:
            camera = renderer.GetActiveCamera()
            camera.SetFocalPoint(origin)
        else:
            renderer.SetActiveCamera(render_camera)

        myvtk.plt_polydata(renderer, scapula_curves, color='cornsilk')
        for p in glenoid_mesh_ids:
            myvtk.plt_point(renderer, scapula_curves.GetPoint(p), radius=0.25, color='black')
        myvtk.plt_point(renderer, initial_point, radius=1, color='red')
        myvtk.plt_point(renderer, origin, radius=1, color='green')
        myvtk.plt_line(renderer, origin, origin + (normal * 10))

        myvtk.show_scene(renderer, save_path=render_save_path)

    if render is True:
        render_mesh_growing()

    return glenoid_plane, glenoid_mesh_ids


def create_glenoid_mask(blank_mask, scapula_mesh, glenoid_mesh_ids, origin, normal, buffer_distance=5,
                        render=False,
                        render_camera=None,
                        render_save_path=None):

    segment_plane = myvtk.vtkPlane()
    segment_plane.SetOrigin(origin[0], origin[1], origin[2])
    segment_plane.SetNormal(normal[0], normal[1], normal[2])

    # create second plane above all glenoid points
    mesh_points = myvtk.get_points(scapula_mesh, glenoid_mesh_ids)
    d_above, _ = myvtk.compute_furthest_distance_from_plane(segment_plane, mesh_points, constraint='above')

    plane_above = myvtk.vtkPlane()
    origin_above = origin + (d_above + buffer_distance) * normal
    plane_above.SetOrigin(origin_above[0], origin_above[1], origin_above[2])
    plane_above.SetNormal(normal[0], normal[1], normal[2])

    # create a 3D convex hull from glenoid surface point projections onto the two planes
    glenoid_shape_points_list = []
    for p in glenoid_mesh_ids:
        point = scapula_mesh.GetPoint(p)
        glenoid_shape_points_list.append(myvtk.project_point_onto_plane(segment_plane, point))
        glenoid_shape_points_list.append(myvtk.project_point_onto_plane(plane_above, point))

    glenoid_shape_points = np.asarray(glenoid_shape_points_list)
    poly = myvtk.points_to_polydata(glenoid_shape_points)
    hull = myvtk.convex_hull(poly)

    # subdivide to ensure there are points along sides of hull
    # for which normals can be computed pointing outward parallel to planes
#    hull = myvtk.subdivide_mesh(hull)

    # extend points outward by a buffer distance and recompute hull
    normals = myvtk.get_point_normals(hull)
    points = myvtk.get_points(hull)
    #glenoid_shape_points_list = []
    for i in range(len(points)):
        normal_tip = points[i] + normals[i]
        normal_tip_proj = myvtk.project_point_onto_plane(segment_plane, normal_tip)
        new_normal = normal_tip_proj - points[i]
        new_normal /= np.linalg.norm(new_normal, 2)

        new_point = points[i] + buffer_distance * new_normal
        glenoid_shape_points_list.append(myvtk.project_point_onto_plane(segment_plane, new_point))
        glenoid_shape_points_list.append(myvtk.project_point_onto_plane(plane_above, new_point))

    glenoid_shape_points = np.asarray(glenoid_shape_points_list)
    poly2 = myvtk.points_to_polydata(glenoid_shape_points)
    glenoid_hull = myvtk.convex_hull(poly2)

    # glenoid_mask = myvtk.extract_image_geometry_from_mesh(blank_mask, hull2)
    glenoid_mask = myvtk.polydata_to_imagedata(glenoid_hull, image_template=blank_mask)

    def render_glenoid_hull(scapula_mesh, glenoid_hull, segment_origin):

        renderer = vtkRenderer()
        if render_camera is None:
            camera = renderer.GetActiveCamera()
            camera.SetFocalPoint(segment_origin)
        else:
            renderer.SetActiveCamera(render_camera)

        myvtk.plt_polydata(renderer, scapula_mesh, color='cornsilk')
        myvtk.plt_polydata(renderer, glenoid_hull, color=(0.5,0.5,0.5,0.5))

        myvtk.show_scene(renderer, save_path=render_save_path)

    if render is True:
        render_glenoid_hull(scapula_mesh, glenoid_hull, origin)

    return glenoid_mask


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
    if len(points) > 3:
        # Find a convex hull in O(N log N)
        hull = ConvexHull(points)

        # Extract the points forming the hull
        hull_points = points[hull.vertices, :]
    else:
        hull_points = points

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
        mid_points[i, :] = (p1 + p2) / 2
    return mid_points



if __name__ == '__main__':
    main()