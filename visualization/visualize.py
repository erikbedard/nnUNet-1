import os.path

import numpy as np
import pyvista as pv
from tqdm import tqdm

from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.util.numpy_support import numpy_to_vtk


def visualize_numpy_mask(mask: np.ndarray, spacing=(1, 1, 1), origin=(0, 0, 0)):
    assert(np.all(np.logical_or(mask == 0, mask == 1)))
    grid = pv.UniformGrid()
    grid.dimensions = np.array(mask.shape) + 1
    grid.spacing = spacing
    grid.origin = origin
    grid.cell_data["values"] = mask.flatten(order="F")

    arr = grid.cell_data["values"]
    cell_ids = np.argwhere(arr == 1)
    grid = grid.extract_cells(cell_ids)
    grid.plot()


def visualize_mask(mask: vtkImageData, scalars: np.ndarray = None, scalar_name="", save_video_path=None, show_plot=True):
    mask = pv.wrap(mask)
    mask_scalars_name = mask.active_scalars_name
    mask = _image_points_to_voxels(mask)

    if scalars is not None:
        mask.cell_data[scalar_name] = scalars.flatten(order="F").astype('float32')
    mask = mask.threshold(value=1, scalars=mask_scalars_name)
    # arr = mask.cell_data["NIFTI"]
    # cell_ids = np.argwhere(arr == 1)
    # mask = mask.extract_cells(cell_ids)

    if show_plot:
        pv.set_plot_theme('document')
        plotter = pv.Plotter()
        plotter.hide_axes()

        if np.all(mask.active_scalars == 1):
            plotter.add_mesh(mask, color='cornsilk')
        else:
            plotter.add_mesh(mask)
            plotter.remove_scalar_bar()
            plotter.add_scalar_bar(title=scalar_name, vertical=True)

        plotter.show()

    if save_video_path is not None:
        render_video(mask, save_video_path)


def _image_points_to_voxels(image: pv.UniformGrid):
    origin = image.origin
    spacing = np.array(image.spacing)
    dims = np.array(image.dimensions) + 1

    m_dims = (3, 3)
    d_matrix = image.GetDirectionMatrix()
    nd_matrix = np.empty(m_dims)
    for i in range(m_dims[0]):
        for j in range(m_dims[1]):
            nd_matrix[i, j] = d_matrix.GetElement(i, j)

    origin = origin - np.matmul(nd_matrix, spacing) / 2

    voxel_img = pv.UniformGrid(dims=dims, spacing=spacing, origin=origin)

    num_arrays = image.GetPointData().GetNumberOfArrays()
    for i in range(num_arrays):
        arr = image.GetPointData().GetAbstractArray(i)
        voxel_img.GetCellData().AddArray(arr)

    voxel_img.set_active_scalars(image.active_scalars_name)

    return voxel_img


def visualize_image_labels(reference_mask: vtkImageData, comparison_mask: vtkImageData, comparison_type="generic", opacities=(1,1,1)):

    def plot_meshes(*meshes, opacities=(1, 1, 1), legend_names=("Intersection", "Added", "Removed")):
        pv.set_plot_theme('document')
        plotter = pv.Plotter()
        colors = ['grey', 'orange', 'blue']

        for i, mesh in enumerate(meshes):
            if mesh.number_of_points > 0:
                plotter.add_mesh(mesh, label=legend_names[i], opacity=opacities[i], color=colors[i])
            plotter.add_legend(bcolor=(0.95,0.95,0.95))

        plotter.show()

    def extract_foreground(mesh: pv.UniformGrid, array_name: str, foreground: float = 1):
        arr = mesh.cell_data[array_name]
        cell_ids = np.argwhere(arr == foreground)
        return mesh.extract_cells(cell_ids)

    def create_meshes(reference_mask, comparison_mask, data_labels):
        reference = pv.wrap(reference_mask)
        comparison = pv.wrap(comparison_mask)
        reference = _image_points_to_voxels(reference)
        comparison = _image_points_to_voxels(comparison)

        reference_array = reference.cell_data["NIFTI"].astype('uint8')
        comparison_array = comparison.cell_data["NIFTI"]

        # create meshes to visualize
        intersection = reference_array * comparison_array
        inter_mesh = pv.UniformGrid()
        inter_mesh.deep_copy(reference)
        inter_mesh.cell_data[data_labels[0]] = intersection

        comparison_not_reference = comparison_array * np.where(reference_array == 0, 1, 0)
        comparison_not_reference_mesh = pv.UniformGrid()
        comparison_not_reference_mesh.deep_copy(reference)
        comparison_not_reference_mesh.cell_data[data_labels[1]] = comparison_not_reference

        reference_not_comparison = reference_array * np.where(comparison_array == 0, 1, 0)
        reference_not_comparison_mesh = pv.UniformGrid()
        reference_not_comparison_mesh.deep_copy(reference)
        reference_not_comparison_mesh.cell_data[data_labels[2]] = reference_not_comparison

        inter_mesh = extract_foreground(inter_mesh, array_name=data_labels[0])
        comparison_not_reference_mesh = extract_foreground(comparison_not_reference_mesh, array_name=data_labels[1])
        reference_not_comparison_mesh = extract_foreground(reference_not_comparison_mesh, array_name=data_labels[2])

        return inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh

    if comparison_type == "generic":
        data_labels = ("Intersection", "Added", "Removed")
    elif comparison_type == "truth_prediction":
        data_labels = ("Correct Prediction", "False Positive", "False Negative")
    else:
        return RuntimeError

    inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh \
        = create_meshes(reference_mask, comparison_mask, data_labels)

    plot_meshes(inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh,
                opacities=opacities,
                legend_names=data_labels)


def render_video(mesh, file_name: str,
                 frame_rate: int = 30,
                 aspect_ratio=np.array([8, 9]),
                 ppi=160,  # leave this as a multiple of 16 or codec fails
                 revolution=360,  # degrees in a full revolution
                 n_frames=360,  # frames per revolution
                 fade_in=0,  # number of frames to hold still for before rotating
                 fade_out=0,  # number of frames to hold still for after rotating
                 scalar_name=""):
    def get_mesh_center(mesh: pv.UnstructuredGrid):
        bnds = np.array(mesh.bounds)  # (xmin, xmax, ymin, ymax, zmin, zmax)
        bnd_lengths = np.array([bnds[1] - bnds[0], bnds[3] - bnds[2], bnds[5] - bnds[4]])
        mesh_center = bnds[[0, 2, 4]] + bnd_lengths / 2
        return mesh_center

    resolution = aspect_ratio * ppi
    mesh = pv.wrap(mesh)
    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.open_movie(file_name, framerate=frame_rate)

    sargs = dict(
        title=scalar_name,
        fmt="%.3f",
        shadow=True,
        vertical=True
    )
    if np.all(mesh.active_scalars == 1):
        plotter.add_mesh(mesh, lighting=True, color="cornsilk")
    else:
        plotter.add_mesh(mesh, lighting=True, scalar_bar_args=sargs)
    plotter.show(auto_close=False)

    for _ in range(fade_in):
        plotter.write_frame()

    angle = revolution / n_frames
    mesh_center = get_mesh_center(mesh)
    plotter.write_frame()
    for _ in tqdm(range(n_frames + 1), desc="rendering '" + file_name + "'"):
        mesh.rotate_z(angle, point=mesh_center, inplace=True)
        plotter.write_frame()

    for _ in range(fade_out):
        plotter.write_frame()

    plotter.close()


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


def apply_mask_to_image(mask: vtkImageData, image: vtkImageData):
    name = image.GetPointData().GetScalars().GetName()
    image_scalars = vtk_to_numpy(image.GetPointData().GetScalars())
    mask_scalars = vtk_to_numpy(mask.GetPointData().GetScalars())
    assert(mask.GetScalarRange() == (0,1))
    applied = image_scalars * mask_scalars

    vtk_scalars = numpy_to_vtk(applied)
    vtk_scalars.SetName(name)
    image.GetPointData().SetScalars(vtk_scalars)
    return image


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