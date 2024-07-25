import os
import numpy as np

import Solver, Loss
from Utils import init_data, save_output, initialize_focus_distances
from Analyze import analyze_loss_per_camera
from ParamsParser import load_params, get_loss_params, get_camera_params

def assignment_step(loss_per_pt, debug=False):
    """
        Fix parameters (focus distances), assign each point to a camera.
        The loss is computed for each point w.r.t. each camera given the current focus distances.

    Args:
        loss_per_pt: For each point, the loss w.r.t. each camera, np array of shape (num points x num cameras)

    Returns:
        pts_set_per_camera: list of point set for each camera. NOTE: Point set is represented in point ids, 1-d np array.
        loss: total loss
    """

    num_cameras = loss_per_pt.shape[1]
    assigned_view_per_point = np.argmin(loss_per_pt, axis=1)
    losses = np.min(loss_per_pt, axis=1) # loss for each point

    # re-arrange to associate a point set to each camera
    pts_set_per_camera = []
    for i in range(num_cameras):
        pts_assigned = np.where(assigned_view_per_point == i)[0]
        if debug:
            # check number of points assigned to each camera
            print("camera {}: {} pts".format(i, len(pts_assigned)))
        pts_set_per_camera.append(pts_assigned)

    loss = np.sum(losses)

    return (pts_set_per_camera, loss)

def optimization_step(pts, pts_normals, pts_set_per_camera, vis_pts_set_per_camera,\
                       camera_positions, camera_dirs, camera_params, loss_params,\
                        solver_type, focus_distances_prev, debug=False):
    """
        Fix assignment, solve for a focus distance for each camera.

    Args:
        pts: total point set, np array of shape (num points x 3) (in cartesian coordinates)
        pts_normals: normals of the total point set, np array of shape (num points x 3)
        pts_set_per_camera: list of point set for each camera. Each point set is represented in point ids, 1-d np array.
                            NOTE: Ids are ordered in the order of the total point set.
                            NOTE: The points are already processed so that all the points are visible to the camera
        vis_pts_set_per_camera: list of visible point set for each camera. Each point set is represented in point ids, 1-d np array.
        camera_positions: array of camera positions, np array of shape (num cameras x 3)
        camera_dirs: array of camera viewing directions, np array of shape (num cameras x 3)
        camera_params: camera parameters, loaded from the params file
        loss_params: loss parameters, loaded from the params file
        solver_type: type of solver to use, "grid" or "adaptive"
        focus_distances_prev: previous focus distances, only used for grid solver

    Returns:
        focus_distances: array of focus distances, np array of shape (num cameras, )
        loss: total loss
    """

    focus_distances = []
    losses = [] # total loss for each camera
    for i, pts_set_ids, vis_pts_ids, camera_position, camera_dir in zip(range(len(camera_positions)),\
            pts_set_per_camera, vis_pts_set_per_camera, camera_positions, camera_dirs):

        # no points assigned to this camera
        if len(pts_set_ids) == 0:  
            if debug:
                print("Warning: camera {} has no points assigned".format(i))
            focus_distances.append(0)
            losses.append(0)
            continue

        pts_set_ids = np.array(pts_set_ids)
        vis_pts_ids = np.array(vis_pts_ids)
        vis_ids = np.arange(len(pts_set_ids))[np.isin(pts_set_ids, vis_pts_ids)] # update vis_ids in pts

        points = pts[pts_set_ids]
        normals = pts_normals[pts_set_ids]

        # TODO: make this outside the loop by passing the functor in
        if solver_type == "grid":
            focus_distance, loss = Solver.solveFocusDistanceGrid(points, normals, vis_ids,\
                                                                camera_position, camera_dir,\
                                                                      camera_params, loss_params, focus_distances_prev[i])
        elif solver_type == "adaptive":
            focus_distance, loss = Solver.solveFocusDistanceAdaptive(points, normals, vis_ids,\
                                                                    camera_position, camera_dir, camera_params, loss_params)
        else:
            raise ValueError("Invalid solver type: ", solver_type)

        # sanity check
        if loss == float("inf"):
            raise ValueError("No focus distance found for camera {}.".format(i))

        losses.append(loss)
        focus_distances.append(focus_distance)

    focus_distances = np.array(focus_distances)
    losses = np.array(losses)
    loss = np.sum(losses)

    return (focus_distances, loss)


def main():

    import argparse
    import shutil
    import time

    # argument set-up
    parser = argparse.ArgumentParser(description="Run EM algorithm for focus distance optimization.")
    parser.add_argument("-i", "--input", type=str, help="Path to subject folder")
    parser.add_argument("-c", "--camera", type=str, help="Name of camera config")
    parser.add_argument("-p", "--params", type=str, help="Path to params file")
    parser.add_argument("-pcd", "--pcd", type=str, default="sampled.ply", help="Name of pcd file")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode (store true)")
    parser.add_argument("-s", "--save_dir", type=str, default="output_em", help="Path to save dir")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output filename")

    # Parse the command line arguments to an object
    args = parser.parse_args()
    if not args.input or not args.camera or not args.params:
        print("No input data is provided.")
        print("For help type --help")
        exit()

    # load data
    root_dir = args.input
    camera_pkl_file = os.path.join(root_dir, args.camera)
    mesh_path = os.path.join(root_dir, "real_scale_in_mm.ply")
    sampled_pts_path = os.path.join(root_dir, args.pcd)
    pts, pts_normals, vis_pts_set_per_camera, cameras_network = init_data(mesh_path, sampled_pts_path, camera_pkl_file)
    camera_positions, camera_dirs, _ = cameras_network.parseCameras()

    DEBUG = args.debug
    SAVE_DIR = os.path.join(args.input, args.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # load params
    param_file = args.params
    params = load_params(param_file)
    print("Params: ", params)
    NUM_ITERS = params['alg']["num_iters"]
    INITIALIZATION = params['alg']["initialization"]
    SOVLER_TYPE = params['alg']["solver"]
    camera_params = get_camera_params(params) 
    loss_params = get_loss_params(params)
    shutil.copy(param_file, os.path.join(SAVE_DIR, "params.yml")) # copy params to the input folder

    pts_set_per_camera_updated = None

    # EM algorithm
    t0 = time.perf_counter()
    for iter in range(NUM_ITERS):

        print("Iteration {}".format(iter))
        if iter == 0:
            # initialize focus distance for each camera
            focus_distances = initialize_focus_distances(pts, vis_pts_set_per_camera,\
                                                        camera_positions, camera_dirs, mode=INITIALIZATION)

            # update loss based on current focus distances: for each pt, compute loss w.r.t. each camera
            loss_per_pt = Loss.getViewQualiyLossAllCameras(pts, pts_normals, vis_pts_set_per_camera,\
                                                           camera_positions, camera_dirs, focus_distances=focus_distances,\
                                                           camera_params=camera_params, loss_params=loss_params)

            # initialize assignment
            pts_set_per_camera_updated, loss_ass_step = assignment_step(loss_per_pt, debug=DEBUG)
            total_loss = loss_ass_step
            print("Total loss for initialization: {}".format(total_loss))

        else:
            #  update loss based on current focus distances: for each pt, compute loss w.r.t. each camera
            loss_per_pt = Loss.getViewQualiyLossAllCameras(pts, pts_normals, vis_pts_set_per_camera,\
                                                           camera_positions, camera_dirs, focus_distances=focus_distances,\
                                                           camera_params=camera_params, loss_params=loss_params)

            ### Assignment step: Fix focus distance, assign each point to a camera
            print("=====Assignemnt step=====")
            pts_set_per_camera_updated, loss_ass_step = assignment_step(loss_per_pt, debug=DEBUG)
            print("Total loss for assignment: {}".format(loss_ass_step))

            ### Optimization step: Fix assignment, solve for a focus distance for each camera
            print("=====Optimization step=====")
            focus_distances_updated, loss_opt_step = optimization_step(pts, pts_normals, pts_set_per_camera_updated,\
                                                                         vis_pts_set_per_camera, camera_positions, camera_dirs,\
                                                                         camera_params, loss_params, SOVLER_TYPE,
                                                                         focus_distances_prev=focus_distances, debug=DEBUG)
            print("Total loss for optimization ({}): {}".format(loss_params['focal_loss_type'], loss_opt_step))

            if DEBUG:
                # sanity check: loss per camera in the optimization step should be at least as small as in the assignmenet step
                loss_per_camera_ass = analyze_loss_per_camera(pts, pts_normals, pts_set_per_camera_updated, vis_pts_set_per_camera,\
                                                                camera_positions, camera_dirs, focus_distances,\
                                                            camera_params=camera_params, loss_params=loss_params)
                loss_per_camera_opt = analyze_loss_per_camera(pts, pts_normals, pts_set_per_camera_updated, vis_pts_set_per_camera,\
                                                            camera_positions, camera_dirs, focus_distances_updated,\
                                                            camera_params=camera_params, loss_params=loss_params)
                loss_per_camera_diff = loss_per_camera_ass - loss_per_camera_opt
                incorrect_cam_ids = np.where(loss_per_camera_diff < -0.0001)[0]
                assert len(incorrect_cam_ids) == 0, "[ERROR]: Loss is not monotonically decreasing for individual cameras"

            # sanity check
            if loss_opt_step - loss_ass_step > 0.001: # could be numerical difference
                print("[WARNING]: loss from optimization step is larger than the loss from assignment step.")
            
            if DEBUG:
                # sanity check: see if any pts assigned are invisible to the camera
                for cam_id, pts_set_ids, vis_pts_ids in zip(range(len(camera_positions)), pts_set_per_camera_updated, vis_pts_set_per_camera):
                    invis_pts_ids = np.arange(len(pts_set_ids))[~np.isin(pts_set_ids, vis_pts_ids)]
                    if len(invis_pts_ids) > 0:
                        print("[WARNING]: camera {} has {} invisible pts assigned:".format(cam_id, len(invis_pts_ids)))

            # termination condition
            if abs(loss_opt_step - total_loss) < 0.01:
                break

            if DEBUG:
                # print out changes in focus distances
                focus_change_thresh = 0.1
                camera_ids = np.where(np.abs(np.array(focus_distances_updated) - np.array(focus_distances)) > focus_change_thresh)[0]
                print("focus distances change in cameras:")
                print(camera_ids)
                print("difference:")
                print(np.array(focus_distances_updated)[camera_ids] - np.array(focus_distances)[camera_ids])

            # update
            total_loss = loss_opt_step
            focus_distances = focus_distances_updated.copy()

        # Save results
        outfname = args.output + "_iter_{}.pkl".format(iter)
        save_output(cameras_network, focus_distances, vis_pts_set_per_camera, pts, pts_normals,\
                    pts_set_per_camera_updated, outfname=outfname, save_dir=SAVE_DIR)
    t1 = time.perf_counter()
    duration = t1 - t0
    print("Total Time: {}sec; {} sec per iteration.".format(duration, duration/(iter+1)))

if __name__ == "__main__":
    main()