import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import myvtk

from vtkmodules.vtkRenderingCore import vtkRenderer


def main():
    filepath = r"C:\erik\data\prepared_datasets\Task601_Scapula\labelsTs\219239.nii.gz"

    # get scapula from image
    labels = myvtk.read_nifti(filepath)
    scapula_label_num = 1
    scapula_mask = myvtk.get_mask_from_labels(labels, scapula_label_num)
    scapula_mesh = myvtk.create_mesh_from_image_labels(labels, scapula_label_num, preserve_boundary=False)

    render_geometry = True
    render_glenoid_mesh = True
    render_segment = True

    optim_breadth = 12
    optim_max_dist = 10
    growth_threshold = 0.1
    min_growth_rate = 0.1
    glenoid_thickness = 10
    segment_radius_scale = 1.4
    plane_origin, plane_normal, cutting_radius = determine_glenoid_segment_parameters(
        scapula_mesh,
        optim_breadth=optim_breadth,
        optim_max_dist=optim_max_dist,
        growth_threshold=growth_threshold,
        min_growth_rate=min_growth_rate,
        glenoid_thickness=glenoid_thickness,
        segment_radius_scale=segment_radius_scale,
        render_geometry=render_geometry,
        render_glenoid_mesh=render_glenoid_mesh)

    glenoid_mask = extract_glenoid(
        scapula_mask,
        origin=plane_origin,
        normal=plane_normal,
        radius=cutting_radius,
        render=render_segment)

    filename = os.path.basename(filepath).split('.')[0]
    save_path = r"C:\erik" + os.path.sep + filename + ".nii.gz"
    myvtk.save_nifti(glenoid_mask, save_path)

    import json
    geom_dict = dict()
    geom_dict['plane_origin'] = list(plane_origin)
    geom_dict['plane_normal'] = list(plane_normal)
    geom_dict['glenoid_thickness'] = glenoid_thickness
    geom_dict['cutting_radius'] = cutting_radius

    json_dict = dict()
    json_dict['id'] = filename
    json_dict['source'] = filepath
    json_dict['geometry'] = geom_dict
    save_path = r"C:\erik" + os.path.sep + filename + ".json"
    with open(os.path.join(save_path), "w") as file:
        json.dump(json_dict, file, indent=4)
    # save_path = r"C:\erik" + os.path.sep + filename + ".stl"
    #myvtk.save_stl(scapula_mesh, save_path)


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

    glenoid_plane, glenoid_diameter = compute_glenoid_plane(scapula_mesh,
                                                            point_on_glenoid,
                                                            optim_breadth=optim_breadth,
                                                            optim_max_dist=optim_max_dist,
                                                            growth_threshold=growth_threshold,
                                                            min_growth_rate=min_growth_rate,
                                                            render=render_glenoid_mesh)

    glenoid_radius = glenoid_diameter/2
    segment_radius = segment_radius_scale * glenoid_radius

    origin = np.asarray(glenoid_plane.GetOrigin())
    normal = np.asarray(glenoid_plane.GetNormal())

    # offset origin
    segment_origin = origin - normal * glenoid_thickness
    segment_normal = normal

    return segment_origin, segment_normal, segment_radius


def get_initial_glenoid_point(scapula_mesh, render=False):
    # goal is to find a seed point on the glenoid

    # find acromion angle (AA) and inferior angle (IA) points by getting furthest two points in the mesh
    scapula_vertices, _ = myvtk.extract_mesh_data(scapula_mesh)
    p1, p2 = find_farthest_points(scapula_vertices)

    # cast a line between both points through the model and find intersections
    intersect_pts = myvtk.ray_casting(scapula_mesh, p1, p2)

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
    ind = np.argmax(distances)
    midpoint_near_glenoid = midpoints[ind]

    # we can now identify the AA as the point that is closer to MP than IA and get a point on glenoid
    d1 = np.linalg.norm(p1 - midpoint_near_glenoid, 2)
    d2 = np.linalg.norm(p2 - midpoint_near_glenoid, 2)
    if d1 < d2:
        lateral_acromion = p1
        inferior_angle = p2
        point_on_glenoid = intersect_pts[-3]  # -1 is coincident, -2 is acromion surface, -3 is glenoid
    else:
        lateral_acromion = p2
        inferior_angle = p1
        point_on_glenoid = intersect_pts[2]  # 0 is coincident, 1 is acromion surface, 2 is glenoid

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

    # calc glenoid diameter
    glenoid_points = []
    for p in glenoid_mesh_ids:
        glenoid_points.append(scapula_mesh.GetPoint(p))
    p1, p2 = find_farthest_points(np.asarray(glenoid_points))
    glenoid_diameter = np.linalg.norm(p1-p2, 2)

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

    return glenoid_plane, glenoid_diameter


def extract_glenoid(scapula_mask, origin, normal, radius, render=False):
    plane = myvtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    scalars = myvtk.get_scalars(scapula_mask)
    scapula_ind = np.argwhere(scalars)
    for i in scapula_ind:
        point = np.asarray(scapula_mask.GetPoint(i))
        d_to_plane = myvtk.compute_signed_distance_to_plane(plane, point)
        wrong_side_of_plane = d_to_plane < 0

        point_is_glenoid = True
        if wrong_side_of_plane:
            point_is_glenoid = False
        else:
            d_to_origin = np.linalg.norm(origin-point, 2)
            too_far_from_origin = d_to_origin > radius
            if too_far_from_origin:
                point_is_glenoid = False

        if point_is_glenoid is False:
            myvtk.set_scalar(scapula_mask, i, 0)

    glenoid_mask = scapula_mask

    def render_glenoid(glenoid_mask, segment_origin):
        foreground = 1
        glenoid_mesh = myvtk.create_mesh_from_image_labels(glenoid_mask, foreground, preserve_boundary=True)

        renderer = vtkRenderer()
        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(segment_origin)
        myvtk.plt_polydata(renderer, glenoid_mesh, color='cornsilk')
        myvtk.show_scene(renderer)

    if render is True:
        render_glenoid(glenoid_mask, origin)

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