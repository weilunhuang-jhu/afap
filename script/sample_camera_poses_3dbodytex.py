import sys, os
sys.path.append("../lib")

import numpy as np
import pyvista as pv
import trimesh
import pickle
import networkx as nx

from Camera import CameraData, Intrinsics, CamerasNetwork

def sample_on_cylinder(mesh, camera_data_template, num_vertical=5, num_angular=50, normal_offset=300, radius=None):

    body_length_offset = mesh.length / 2 - 200
    center_p = np.array(mesh.center)
    normal_p = np.array([0,1,0]) # positive y-axis
    center_p = center_p - normal_p * body_length_offset
    _, H = trimesh.points.project_to_plane(center_p.reshape((-1,3)), plane_normal=normal_p, plane_origin=center_p, return_planar=False, return_transform=True)

    camera_positions = []
    camera_poses = []
    cameras = []
    # create cirular trajectories along the cylinder
    for i in range(num_vertical):
        center = center_p + i * normal_p * normal_offset
        for j in range(num_angular):
            theta = 2*np.pi*j/num_angular
            x = radius*np.cos(theta)
            y = radius*np.sin(theta)
            z = 0
            pos = np.matmul(H, np.array([x,y,z,1]).reshape((-1,1)))
            pos = pos.flatten()[:3]
            # offset the circle along the normal direction
            pos = pos + i * normal_p * normal_offset
            # create a pose
            z_dir = center - pos # view_dir
            z_dir = z_dir / np.linalg.norm(z_dir)
            y_dir = normal_p # -up_dir
            x_dir = np.cross(y_dir, z_dir)
            x_dir = x_dir / np.linalg.norm(x_dir)
            pose = np.eye(4)
            pose[:3,0] = x_dir
            pose[:3,1] = y_dir
            pose[:3,2] = z_dir
            pose[:3,3] = pos

            camera_data = CameraData(None, None, None)
            camera_data.fname = "camera_{}_{}".format(i, str(j).zfill(3))
            camera_data.img_resolution = camera_data_template.img_resolution
            camera_data.intrinsics = camera_data_template.intrinsics
            camera_data.center = pos
            camera_data.orientation = pose[:3,:3]
            camera_data.pose = pose

            cameras.append(camera_data)
            camera_poses.append(pose)
            camera_positions.append(pos)

    camera_positions = np.array(camera_positions)
    camera_poses = np.array(camera_poses)

    return (cameras, camera_positions, camera_poses)

def create_camera_topology_cylinder(num_vertical=5, num_angular=50):
    # create topology for camera poses on a cylinder
    # TODO: create topology with the edge weighted by 
    #   1) the distance between the camera positions
    #   2) the overlap of FOV of the cameras
    #
    #                 cam - cam - cam
    #                  |     |     |
    #                 cam - cam - cam
    #                  |     |     |
    #                 cam - cam - cam

    G = nx.grid_2d_graph(num_vertical, num_angular, periodic=(False, True))
    labels = {}
    for i in range(num_vertical):
        for j in range(num_angular):
            labels[(i,j)] = i*num_angular + j
    G = nx.relabel_nodes(G, labels) # relabel nodes to 0, 1, 2, ...

    return G

def main():

    import argparse

    # argument set-up
    parser = argparse.ArgumentParser(description="Create synthetic camera poses on a cylinder.")
    parser.add_argument("-i", "--input", type=str, help="path to input data folder")
    parser.add_argument("-z", "--num_vertical", type=int, help="Number of vertical samples")
    parser.add_argument("-a", "--num_angular", type=int, help="Number of angular samples")
    parser.add_argument("-r", "--radius", type=int, help="Radius of the cylinder (mm)")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode (store true)")

    # # Parse the command line arguments to an object
    args = parser.parse_args()
    if not args.input or not args.num_vertical or not args.num_angular or not args.radius:
        parser.print_help()
        sys.exit(0)

    DEBUG = args.debug
    SAVE_DIR = args.input
    os.makedirs(SAVE_DIR, exist_ok=True)

    root_dir = args.input
    # load mesh data
    mesh_path = os.path.join(root_dir, "real_scale_in_mm.ply")
    texture_path = os.path.join(root_dir, "model_lowres_0_normalized.png")
    texture_pv = pv.read_texture(texture_path)
    mesh_pv = pv.read(mesh_path)

    # camera parameters
    camera_intrinsics = Intrinsics(fx=1190.7321488962878, fy=1190.7321488962878, px=284.81279922958316, py=434.4404582539672, dist=[0,0,0,0,0]) # calibrated from Canon 90D
    camera_data_template = CameraData(fname="", img_resolution=(580,870), intrinsics=camera_intrinsics)

    # cylinder parameters
    total_height = 1600 # use the same total height to account for the same volume
    num_vertical = args.num_vertical
    num_angular = args.num_angular
    normal_offset = total_height // (num_vertical - 1)
    radius = args.radius
    cameras_config = (num_vertical, num_angular, normal_offset, radius)
    cameras, camera_positions, camera_poses = sample_on_cylinder(mesh_pv, camera_data_template=camera_data_template,\
                                                num_vertical=num_vertical, num_angular=num_angular,\
                                                normal_offset=normal_offset, radius=radius)
    cameras_topology = create_camera_topology_cylinder(num_vertical=num_vertical, num_angular=num_angular)
    cameras_network = CamerasNetwork(cameras_config, cameras, cameras_topology)

    outfname = "camera_poses_cylinder" + "_{}_{}_{}_{}.pkl".format(num_vertical, num_angular, normal_offset, radius)
    outfname = os.path.join(SAVE_DIR, outfname)
    pickle.dump(cameras_network, open(outfname, "wb"))

    #  Visaulization 
    if DEBUG:

        np.random.seed(0)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_pv, name="mesh", texture=texture_pv)

        axes_length = 50
        for tf in camera_poses:
            center = tf[:3,3]
            x = tf[:3,0]
            y = tf[:3,1]
            z = tf[:3,2]
            # Visualize camera 
            plotter.add_arrows(center, x, mag=axes_length, color='Red')
            plotter.add_arrows(center, y, mag=axes_length, color='Green')
            plotter.add_arrows(center, z, mag=axes_length, color='Blue')
            camera_sphere = pv.Sphere(radius=20, center=center)
            plotter.add_mesh(camera_sphere, color="white")
        plotter.show_axes()
        plotter.reset_camera()
        plotter.show()

if __name__ == "__main__":
    main()