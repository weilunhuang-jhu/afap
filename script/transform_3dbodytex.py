import os
import pandas as pd
import numpy as np
import trimesh
import pymeshlab
import argparse

# Script to transform the 3dbodytex data by: (for convenience since the parameters used in this work is in mm) 
# NOTE: The script also transform the data to align with the convention used in [3D-CODED](https://github.com/ThibaultGROUEIX/3D-CODED).

# argument set-up
parser = argparse.ArgumentParser(description="Transform 3dbodytex data (lowres)")
parser.add_argument("-i", "--input", type=str, help="Input directory of the 3dbodytex data")
parser.add_argument("-o", "--output", type=str, default="../data/3dbodytex", help="Output folder of the transformed 3dbodytex data")

# Parse the command line arguments to an object
args = parser.parse_args()
if not args.input:
    print("No input folder is provided.")
    print("For help type --help")
    exit()

output_dir = args.output # path to output folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mesh_basename = "real_scale_in_mm.ply"
unit = "mm"
scale_factor = 1
if unit == "mm":
    scale_factor = 1000

root_dir = args.input
scan_names = os.listdir(root_dir)
scan_names.sort()

for scan_name in scan_names:
    scan_id = scan_name.split("-")[0]
    print(scan_id)

    # load data     
    obj_filename = os.path.join(root_dir, scan_name, "model_lowres_0_normalized.obj")
    mesh_trimesh = trimesh.load(obj_filename, process=False) # still has re-ordering because of wedge texture
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_filename)
    
    # apply transformation
    ms.compute_matrix_from_translation(axisx=-mesh_trimesh.centroid[0], axisy=-mesh_trimesh.centroid[1],\
                                        axisz=-mesh_trimesh.centroid[2], freeze=False)                
    ms.compute_matrix_from_scaling_or_normalization(axisx=scale_factor, axisy=scale_factor, axisz=scale_factor, freeze=False)
    ms.compute_matrix_from_rotation(rotaxis=2, angle=90, freeze=False)

    # output dir per scan
    output_dir_scan = os.path.join(output_dir, scan_id)
    if not os.path.exists(output_dir_scan):
        os.makedirs(output_dir_scan)

    # save transformation matrix
    transformation_matrix = ms.current_mesh().trasform_matrix()
    output_fname_tf = os.path.join(output_dir_scan, "transformation.txt")
    np.savetxt(output_fname_tf, transformation_matrix, fmt='%s', delimiter=',')
    
    # freeze transformation for the mesh
    ms.apply_matrix_freeze()
    
    # save ply
    output_fname_ply = os.path.join(output_dir_scan, mesh_basename)
    ms.save_current_mesh(output_fname_ply)