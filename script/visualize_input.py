import sys, os
sys.path.append("../lib")
import numpy as np
import pickle
import argparse
import open3d as o3d
import pyvista as pv
from PyQt5.QtWidgets import (QApplication, QFileDialog)

import Camera

point_size_small = 1

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
parser.add_argument("-c", "--camera", type=str, help="Name of the camera network file (.pkl)")

# Parse the command line arguments to an object
args = parser.parse_args()
if not args.input:
    print("No input data is provided.")

root_dir = args.input
camera_pkl_file = os.path.join(root_dir, args.camera)
mesh_path = os.path.join(root_dir, "real_scale_in_mm.ply")
texture_path = os.path.join(root_dir, "model_lowres_0_normalized.png")
texture_pv = pv.read_texture(texture_path)
mesh_pv = pv.read(mesh_path)
sampled_pts_path = os.path.join(root_dir, "sampled.ply")

cameras_network = Camera.loadCamerasNetwork(camera_pkl_file)
cameras = cameras_network.cameras
camera_positions, camera_dirs, camera_poses = cameras_network.parseCameras()

#  Visaulization 
np.random.seed(0)
plotter = pv.Plotter()

# point set 1: points uniformly sampled from mesh
plotter.add_mesh(mesh_pv, name="mesh", texture=texture_pv)
pcd = o3d.io.read_point_cloud(sampled_pts_path)
sampled_pts = np.asarray(pcd.points)
plotter.add_points(sampled_pts, color="gray", point_size=point_size_small, name="sampled_pts")

axes_length = 75
radius = 10

for camera_data, camera_pose in zip(cameras, camera_poses):
    
    device_name = camera_data.fname
    center = camera_pose[:3,3]
    x = camera_pose[:3,0]
    y = camera_pose[:3,1]
    z = camera_pose[:3,2]

    # Visualize camera 
    plotter.add_arrows(center, x, mag=axes_length, color='Red', name='x_'+device_name)
    plotter.add_arrows(center, y, mag=axes_length, color='Green', name='y_'+device_name)
    plotter.add_arrows(center, z, mag=axes_length, color='Blue', name='z_'+device_name)
    camera_sphere = pv.Sphere(radius=radius, center=center)
    plotter.add_mesh(camera_sphere, name=device_name, color="black")

plotter.show_axes()
plotter.background_color = 'white'
plotter.reset_camera()
plotter.add_key_event("s", key_callback_save_camera)
plotter.add_key_event("o", key_callback_load_camera)

plotter.show()