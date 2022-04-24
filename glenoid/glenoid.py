import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import myvtk

from vtkmodules.vtkRenderingCore import vtkRenderer


def main():
    filepath = r"C:\erik\data\nnUNet\predictions\Task600\pred_imagesTr\final_output\ACD.nii.gz"
    label_num = 1  # scapula label

    labels = myvtk.read_nifti(filepath)
    scapula_label_num = 1
    humerus_label_num = 2

    # # extract binary masks
    # np_labels = myvtk.convert_image_to_numpy(labels)
    # scapula = (np_labels == scapula_label_num).astype('uint8')
    # humerus = (np_labels == humerus_label_num).astype('uint8')

    # create scapula mesh
    scapula_mask = myvtk.get_mask_from_labels(labels, scapula_label_num)
    scapula_mesh = myvtk.convert_voxels_to_poly(scapula_mask, method='flying_edges')
    scapula_mesh = myvtk.decimate_polydata(scapula_mesh, target=10000)
    scapula_mesh = myvtk.smooth_polydata(scapula_mesh, n_iterations=30)

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
        closest_points[i], distances[i] = myvtk.find_closest_point(query, scapula_mesh)

    # the midpoint (MP) which is furthest from the scapula must lie between glenoid and acromion
    ind = np.argmax(distances)
    midpoint_near_glenoid = midpoints[ind]

    # we can now identify the AA as the point that is closer to MP than IA and get a point on glenoid
    d1 = np.linalg.norm(p1 - midpoint_near_glenoid, 2)
    d2 = np.linalg.norm(p2 - midpoint_near_glenoid, 2)
    if d1 < d2:
        acromion_angle = p1
        inferior_angle = p2
        point_on_glenoid = intersect_pts[-3]  # -1 is coincident, -2 is acromion surface, -3 is glenoid
    else:
        acromion_angle = p2
        inferior_angle = p1
        point_on_glenoid = intersect_pts[2]  # 0 is coincident, 1 is acromion surface, 2 is glenoid


    filename = os.path.basename(filepath).split('.')[0]
    save_path = r"C:\erik" + os.path.sep + filename + ".stl"
    myvtk.save_stl(save_path, scapula_mesh)

    renderer = vtkRenderer()
    myvtk.plt_polydata(renderer, scapula_mesh, color='cornsilk')
    myvtk.plt_point(renderer, midpoint_near_glenoid, radius=1, color='black')
    myvtk.plt_point(renderer, point_on_glenoid, radius=1, color='red')
    myvtk.plt_point(renderer, acromion_angle, radius=1, color='black')
    myvtk.plt_point(renderer, inferior_angle, radius=1, color='black')
    myvtk.plt_line(renderer, p1, p2, color='red')
    myvtk.show_scene(renderer)


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