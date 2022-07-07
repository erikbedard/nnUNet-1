import argparse
import visualize
import numpy as np
import os
import sys
import pyvista as pv

from vtkmodules.vtkCommonDataModel import vtkImageData
from tqdm import tqdm


def main():

    # EXAMPLE command-line inputs:
    # To visualize a binary mask NIFTI image where 0 is background, 1 is foreground:
    # python visualize_label.py PATH_TO_NIFTI_MASK

    # To visualize a segmentation label (e.g. labels==2):
    # python visualize_label.py PATH_TO_NIFTI_LABELS --label_num 2

    # To visualize a binary mask with scalar values:
    # python visualize_label.py PATH_TO_NIFTI_MASK --scalars_image PATH_TO_NPZ_SCALARS

    # To visualize a binary mask with a sub-region mask applied:
    # python visualize_label.py PATH_TO_NIFTI_MASK --submask_image PATH_TO_NIFTI_SUBMASK

    # Super command with all options:
    # To visualize a segmentation label (array name "NIFTI") with scalars (array name "softmax") and submask
    # (array name "NIFTI") applied, and to label the scalarbar values as "mean", and to save the visualization as a
    # video in the background:
    # python visualize_label.py PATH_TO_NIFTI_IMAGE --labels_array NIFTI --label_num 1 --scalars_image PATH_TO_NPZ_SCALARS --scalars_array softmax --scalars_name mean --submask_image PATH_TO_NIFTI_SUBMASK --submask_array NIFTI --save_video_dir OUTPUT_DIR

    parser = argparse.ArgumentParser()

    # parse labels
    parser.add_argument(dest="labels_image", type=str,
                        help="File path of NIFTI image with label data.")
    parser.add_argument("--labels_array", type=str, default=None,
                        help="Name of array to be read from 'labels_image' for visualization."
                             "If not specified, the default active array is used.")
    parser.add_argument("--label_num", type=int, default=1,
                        help="Integer value of the label to be visualized. "
                             "If not specified, a default value of 1 is used.")

    # optional submask
    parser.add_argument("--submask_image", type=str, default=None,
                        help="Optionally apply a mask to the labels image by specifying a path to a NIFTI file with "
                             "the mask image. ")
    parser.add_argument("--submask_array", type=str, default=None,
                        help="Name of array to be read from 'sub_mask_image'. "
                             "If not specified, the default active array is used.")

    # optional scalars
    parser.add_argument("--scalars_image", type=str, default=None,
                        help="Optionally apply scalar values to the labels image by specifying a path to a NPZ file "
                             "containing the scalar values.")
    parser.add_argument('--scalars_array', type=str, default=None,
                        help="Name of array to be read from 'scalars_image'. "
                             "If not specified, the first variable in the file is used.")
    parser.add_argument("--scalars_name", type=str, default="",
                        help="Name of scalar data to show on the scalar bar. "
                             "If not specified, the name is blank.")

    # optional video
    parser.add_argument("--save_video", default=False, action='store_true',
                        help="Optionally save a 360-degree video of the visualization "
                             "in the same directory as the labels_image.")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Optionally specify directory to save a 360-degree video of the visualization.")
    parser.add_argument("--background", default=False, action='store_true', required = '--save_video' in sys.argv,
                        help="Save video in background only and do not show the plot.")

    args = parser.parse_args()

    # get mask from labels
    labels = visualize.read_nifti(args.labels_image)
    if args.labels_array is not None:
        pv_labels = pv.wrap(labels)
        pv_labels.set_active_scalars(args.labels_array)

    mask = visualize.get_mask_from_labels(labels, args.label_num)

    # apply submask if specified
    if args.submask_image is not None:
        submask = visualize.read_nifti(args.submask_image)
        if args.submask_array is not None:
            pv_submask = pv.wrap(submask)
            pv_submask.set_active_scalars(args.submask_array)

        mask = visualize.apply_mask_to_image(submask, mask)

    # process scalars
    scalars_name = args.scalars_name
    if args.scalars_image is not None:
        np_loaded = np.load(args.scalars_image)
        if args.scalars_array is not None:
            # load specified variable
            scalars = np_loaded[args.scalars_array]
        else:
            # load first variable in list
            scalars = np_loaded[np_loaded.files[0]]

        # check shape/dimensions
        mask_shape = mask.GetDimensions()
        if scalars.ndim == len(mask_shape) + 1:
            # assume smallest dim is for the class
            class_axis = int(np.argmin(scalars.shape))
            scalars = np.take(scalars, args.label_num, axis=class_axis)
        elif scalars.ndims == len(mask_shape):
            # do nothing
            scalars = scalars
        else:
            raise Exception("Scalars do not match dimensions of labels.")

        # swap axis if needed
        if scalars.shape != mask_shape:
            scalars = np.swapaxes(scalars, 0, 2)  # ZYX -> XYZ
            if scalars.shape != mask_shape:
                raise Exception("Scalars do not match dimensions of labels.")

    else:
        scalars = None

    # set video output path
    base = os.path.basename(args.labels_image)[:-7]
    if scalars_name == "":
        video_name = base + "_" + str(args.label_num) + ".mp4"
    else:
        video_name = base + "_" + str(args.label_num) + "_" + scalars_name + ".mp4"

    if args.video_dir is not None:
        # save video to specified dir
        video_path = os.path.join(args.video_dir, video_name)
    elif args.save_video:
        # save video to same dir as input file
        parent_dir = os.path.dirname(args.labels_image)
        video_path = os.path.join(parent_dir, video_name)
    else:
        video_path = None
    show_plot = not args.background

    visualize_mask(mask,
                   scalars=scalars,
                   scalar_name=scalars_name,
                   save_video_path=video_path,
                   show_plot=show_plot)


def visualize_mask(mask: vtkImageData, scalars: np.ndarray = None, scalar_name="", save_video_path=None, show_plot=True):
    mask = pv.wrap(mask)
    mask_scalars_name = mask.active_scalars_name
    mask = visualize.image_points_to_voxels(mask)

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


def render_video(mesh, file_name: str,
                 frame_rate: int = 30,
                 aspect_ratio=np.array([8, 9]),
                 ppi=160,  # leave this as a multiple of 16 or codec fails
                 revolution=360,  # degrees in a full revolution
                 n_frames=360,  # frames per revolution
                 fade_in=0,  # number of frames to hold still for before rotating
                 fade_out=0,  # number of frames to hold still for after rotating
                 scalar_name=""):

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
    mesh_center = visualize.get_mesh_center(mesh)
    plotter.write_frame()
    for _ in tqdm(range(n_frames), desc="rendering '" + file_name + "'"):
        mesh.rotate_z(angle, point=mesh_center, inplace=True)
        plotter.write_frame()

    for _ in range(fade_out):
        plotter.write_frame()

    plotter.close()


if __name__ == "__main__":
    main()