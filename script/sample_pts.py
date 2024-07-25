import sys, os
sys.path.append("../lib")

import open3d as o3d
import argparse

# argument set-up
parser = argparse.ArgumentParser(description="Create synthetic camera poses on a cylinder.")
parser.add_argument("-i", "--input", type=str, help="path to input data folder")
parser.add_argument("-d", "--debug", action="store_true", help="Debug mode (store true)")
parser.add_argument("-n", "--name", type=str, default="sampled.ply", help="Name of the sampled pcd, default: sampled.ply")
parser.add_argument("-p", "--pts", type=int, default=1000, help="Number of sampling points, default: 1,000")
# # Parse the command line arguments to an object
args = parser.parse_args()
if not args.input:
    parser.print_help()
    sys.exit(0)

root_dir = args.input
mesh_path = os.path.join(root_dir, "real_scale_in_mm.ply")
num_pt_samples = args.pts
pcd = o3d.io.read_triangle_mesh(mesh_path)
if not pcd.has_vertex_normals():
    pcd.compute_vertex_normals()
    print("Compute pcd vertex normals:", pcd.has_vertex_normals())

pcd = pcd.sample_points_uniformly(number_of_points=num_pt_samples)
pcd_fname = os.path.join(root_dir, args.name)
o3d.io.write_point_cloud(pcd_fname, pcd)

# test loading sampled points
if args.debug:
    pcd = o3d.io.read_point_cloud(pcd_fname)
    print(pcd)
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
