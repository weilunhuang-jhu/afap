import sys, os
sys.path.append("../lib")
import numpy as np
import glob
import argparse
import pickle
import open3d as o3d
import pyvista as pv
import trimesh
import matplotlib
from PyQt5.QtWidgets import (QApplication, QFileDialog)

import Camera
from Analyze import analyze_loss_per_point
from Utils import load_output, getVisiblePointSetAllCameras
from ParamsParser import load_params, get_loss_params, get_camera_params

def key_callback_save_camera(*args):
    app = QApplication([])
    filename, filetype = QFileDialog.getSaveFileName(None, 'save file', os.getcwd(),
                                'All Files (*);;PKL Files (*.pkl)')
    cam_file = open(filename,'wb')
    cam_params = {}
    cam_params["position"] = plotter.camera.position
    cam_params["focal_point"] = plotter.camera.focal_point
    cam_params["view_up"] = plotter.camera.up
    cam_params["view_angle"] = plotter.camera.view_angle
    cam_params["clipping_range"] = plotter.camera.clipping_range

    print(cam_params)
    pickle.dump(cam_params, cam_file)
    print("cam parmas saved in " + filename)

def key_callback_load_camera(*args):
    app = QApplication([])
    filename, filetype = QFileDialog.getOpenFileName(None, 'open camera file', os.getcwd(),
                                'All Files (*);;PKL Files (*.pkl)')
    cam_file = open(filename,'rb')
    cam_params = pickle.load(cam_file)

    plotter.camera.position = cam_params["position"]
    plotter.camera.focal_point = cam_params["focal_point"] 
    plotter.camera.up = cam_params["view_up"]
    plotter.camera.view_angle = cam_params["view_angle"]
    plotter.camera.clipping_range = cam_params["clipping_range"]
    plotter.update()

    print(cam_params)
    print("cam parmas loaded from " + filename)

# argument set-up
parser = argparse.ArgumentParser(description="Visualize the input")
parser.add_argument("-i", "--input", type=str, help="Path to input folder")
parser.add_argument("-m", "--method", type=str, default="em", help="em or kview")
parser.add_argument("-iter", "--iter", type=int, default=-1, help="Iteration for visualization, default: -1 (last)")

# Parse the command line arguments to an object
args = parser.parse_args()
if not args.input:
    print("No input data is provided.")

# params for visualization
radius_camera = 20 # mm
radius_focus = 10 # mm
axes_length = 100 # mm
point_size_small = 3
point_size_large = 10

# input args
root_dir = args.input
mesh_path = os.path.join(root_dir, "real_scale_in_mm.ply")
mesh_trimesh = trimesh.load(mesh_path, process=False, use_embree=True)
sampled_pts_path = os.path.join(root_dir, "sampled.ply")

# load output data
method = args.method
outfnames = glob.glob(os.path.join(root_dir, "output_"+method, "*.pkl"))
outfnames.sort()
for file in outfnames.copy():
    basename = os.path.basename(file)
    if "output_iter" not in basename:
        outfnames.remove(file)
        continue
outfname = outfnames[args.iter] # NOTE: iter 0 is the initialization, -1 is the last iteration
print("Visualizing {}".format(outfname))

# load params
param_file = os.path.join(os.path.join(root_dir, "output_"+method), "params.yml")
params = load_params(param_file)
camera_params = get_camera_params(params) 
loss_params = get_loss_params(params)
print(params)

cameras_network, focus_distances, max_pts_set_per_camera, pts, pts_normals, pts_set_per_camera = load_output(outfname)
pts_set_per_camera = np.array(pts_set_per_camera, dtype='object') # debug
print("=====================================")
print("Number of visible points: {}".format(len(pts)))
print("=====================================")

cameras = cameras_network.cameras
camera_positions, camera_dirs, camera_poses = cameras_network.parseCameras()

# get loss per point
loss_per_point = analyze_loss_per_point(pts, pts_normals, pts_set_per_camera, max_pts_set_per_camera,\
                                camera_positions, camera_dirs, focus_distances,\
                                camera_params=camera_params, loss_params=loss_params)

#  Visaulization 
np.random.seed(0)
plotter = pv.Plotter()

# interpolate color based on loss for each point
color_b = np.array([1, 0, 0])
color_g = np.array([0, 1, 0])
point_colors = loss_per_point.reshape((-1,1)) * color_b.reshape((1,3)) + (1 - loss_per_point).reshape((-1,1)) * color_g.reshape((1,3))
c_map = matplotlib.colormaps['coolwarm']

# get invisible points
pcd = o3d.io.read_point_cloud(sampled_pts_path)
sampled_pts = np.asarray(pcd.points)
_, pts_set_ids_visible = getVisiblePointSetAllCameras(mesh_trimesh, sampled_pts, cameras, camera_positions, camera_poses, return_unique_set=True)
pts_set_ids_invisible = np.setdiff1d(np.arange(len(sampled_pts)), pts_set_ids_visible)

sargs = dict(
    title="Focus Distance Cost",
    shadow=True,
    color = 'black'
)
# visualize all pts
plotter.add_points(sampled_pts, color="gray", point_size=point_size_small, name="sampled_pts")
# visualize invisible points
plotter.add_points(sampled_pts[pts_set_ids_invisible], color="red", point_size=point_size_large,\
                    render_points_as_spheres=True, name="invisible_pts")
# visualize all points with quality
plotter.add_points(pts, point_size=point_size_large, render_points_as_spheres=True, scalars=loss_per_point, clim=[0,1],\
                    cmap=c_map, scalar_bar_args=sargs, show_scalar_bar=True, name="pts")

total_loss = loss_per_point.sum() + len(pts_set_ids_invisible) * 1.0
plotter.add_text("Total cost: {:.2f}".format(total_loss), position='upper_edge', font_size=15, color='black')
print("Cost: {}".format(total_loss))

plotter.show_axes()
plotter.background_color = 'white'
plotter.reset_camera()
plotter.add_key_event("s", key_callback_save_camera)
plotter.add_key_event("o", key_callback_load_camera)

plotter.show()