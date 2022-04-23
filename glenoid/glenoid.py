import os
import numpy as np
from stl import mesh
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import myvtk

from vtkmodules.vtkRenderingCore import vtkRenderer


def main():
    filepath = r"C:\erik\data\nnUNet\predictions\Task600\pred_imagesTr\final_output\ACD.nii.gz"
    label_num = 1  # scapula label

    labels = myvtk.read_nifti(filepath)
    np_labels = myvtk.convert_image_to_numpy(labels)

    # extract binary masks
    scapula_label_num = 1
    humerus_label_num = 2
    # scapula = (np_labels == scapula_label_num).astype('uint8')
    # humerus = (np_labels == humerus_label_num).astype('uint8')

    # create scapula mesh
    scapula_mask = myvtk.get_mask_from_labels(labels, scapula_label_num)
    scapula_mesh = myvtk.convert_voxels_to_poly(scapula_mask, method='flying_edges')
    scapula_mesh = myvtk.decimate_polydata(scapula_mesh, reduction=0.5)
    scapula_mesh = myvtk.smooth_polydata(scapula_mesh, n_iterations=30)

    # create humerus mesh
    humerus_mask = myvtk.get_mask_from_labels(labels, humerus_label_num)
    humerus_mesh = myvtk.convert_voxels_to_poly(humerus_mask, method='flying_edges')
    humerus_mesh = myvtk.decimate_polydata(humerus_mesh, reduction=0.5)
    humerus_mesh = myvtk.smooth_polydata(humerus_mesh, n_iterations=30)

    # get initial scapula landmarks
    scapula_vertices, _ = myvtk.extract_mesh_data(scapula_mesh)
    humerus_vertices, _ = myvtk.extract_mesh_data(humerus_mesh)
    p1, p2 = get_farthest_points(scapula_vertices)
    acromion_angle, inferior_angle = identify_points(p1, p2, humerus_vertices)

    # visualize scapula
    # vertices, faces = myvtk.extract_mesh_data(scapula_mesh)
    # visualize_mesh(vtk_mesh)

    intersect = myvtk.ray_casting(scapula_mesh, acromion_angle, inferior_angle)
    glenoid_seed_point = intersect[2]  # 0 is coincident, 1 is acromion surface, 2 is glenoid
    distances = np.linalg.norm(intersect-acromion_angle, 2, axis=1)

    filename = os.path.basename(filepath).split('.')[0]
    save_path = r"C:\erik" + os.path.sep + filename + ".stl"
    myvtk.save_stl(save_path, scapula_mesh)

    renderer = vtkRenderer()
    myvtk.plt_polydata(renderer, scapula_mesh, color='cornsilk')
    myvtk.plt_point(renderer, acromion_angle, radius=1, color=[1.0, 0.0, 0.0])
    myvtk.plt_point(renderer, inferior_angle, radius=1, color=[0.0, 1.0, 0.0])
    myvtk.plt_line(renderer, acromion_angle, inferior_angle)
    myvtk.plt_point(renderer, glenoid_seed_point, radius=1, color=[0.0, 1.0, 0.0])
    myvtk.show_scene(renderer)


def identify_points(p1, p2, humerus_points):
    ''' Given two points (p1, p2) which define the diameter of the Scapula,
    use the humerus to determine which of the two points is superior to the other'''
    # Find a convex hull in O(N log N)
    hull = ConvexHull(humerus_points)

    # Extract the points forming the hull
    hull_points = humerus_points[hull.vertices, :]

    d1 = get_shortest_distance(p1, hull_points)
    d2 = get_shortest_distance(p2, hull_points)

    # superior point is the one closest to humerus
    if d1 < d2:
        acromion_angle = p1
        inferior_angle = p2
    else:
        acromion_angle = p2
        inferior_angle = p1

    return acromion_angle, inferior_angle


def get_farthest_points(points):
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


def get_shortest_distance(query_point, target_points):
    hdist = cdist(query_point[np.newaxis, :], target_points, metric='euclidean')

    # Get the closest points
    closest = np.unravel_index(hdist.argmin(), hdist.shape)
    closest_point = target_points[closest[0]]
    return np.linalg.norm(query_point-closest_point, 2)





def visualize_mesh(faces, vertices, color=(0.7, 0.3, 0.3)):


    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces))

    # improve visualization with normals and color
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color(color)

    o3d.visualization.draw_geometries([o3d_mesh])


def save_stl(save_path, faces, vertices):
    """

    Args:
        save_path: directory where file will be saved
        faces: Nx3 np array
        vertices: Nx3 np array
    """
    # Create the mesh
    stl_obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_obj.vectors[i][j] = vertices[f[j], :]

    # Write the mesh to file
    stl_obj.save(save_path)


if __name__ == '__main__':
    main()