import os
import numpy as np
import pickle
import time
import logging
from multiprocessing import Pool, Manager

import Solver
from Utils import wrap_em_common_result, get_em_common_result, load_output, save_output
from Analyze import analyze_loss_per_camera
from ParamsParser import load_params, get_loss_params, get_camera_params
from Logger import setup_logging, log_args

def merge_pts_set(pts_set_per_camera, camera_ids):
    """ Merge point set for a group of cameras.

    Args:
        pts_set_per_camera (_type_): list of point set for each camera. Each point set is represented in point ids, 1-d np array.
        camera_ids (_type_): list of camera ids

    Returns:
        pts_set_ids (np array): merged point set
    """
    # merge point set
    pts_set_ids = np.array([], dtype=int)
    for camera_id in camera_ids:
        logging.debug("{} pts for camera {}".format(len(pts_set_per_camera[camera_id]), camera_id))
        pts_set_ids = np.concatenate((pts_set_ids, pts_set_per_camera[camera_id])).astype(int)
    return pts_set_ids

def optimize_group_views(camera_ids, pts_set_ids, pts, pts_normals, vis_pts_set_per_camera,\
                        camera_positions_group, camera_dirs_group, camera_params, loss_params):
    """
        Optimize the focus distances for a group of cameras:
        - exclude points that are only visible to one camera (the assignment for these points is trivial)
        - solve the foucs optimization problem

        Args:
            camera_ids:
            pts_set_ids: in raw pts id
            pts: 
            pts_normals:
            vis_pts_set_per_camera: in range 0 - len(pts)
            camera_positions_group:
            camera_dirs_group:
            camera_params
            loss_params
    """

    # initialize
    focus_distance = [0] * len(camera_positions_group)
    loss = float('inf')
    assignment = np.ones(len(pts)) * -1

    logging.debug("Num of pts: {}".format(len(pts)))

    # trivial case 1:
    if len(pts_set_ids) == 0: # no points assigned to this camera group
        logging.debug("[Trivial Case]: no points assigned to this camera group.")
        return (focus_distance, loss, assignment)

    # solve: NOTE: assignment is in the order of (0 - number of cameras - 1)
    focus_distance, loss, assignment = Solver.solveFocusDistanceMultiViewsAdaptive(pts, pts_normals, vis_pts_set_per_camera,\
                                        camera_positions_group, camera_dirs_group, camera_params, loss_params)

    # Sanity check: see if points assigned to the cameras are visible (in the maximum set of points)
    for i, vis_pts_set in enumerate(vis_pts_set_per_camera):
        pts_ids = np.where(assignment==i)[0] # in range 0-len(pts)
        num_pts = len(pts_ids)
        num_visible_pts = np.in1d(pts_ids, vis_pts_set).sum()
        logging.debug("{} pts assigned to camera {}.".format(num_visible_pts, camera_ids[i]))
        # assert np.in1d(pts_ids, max_pts_set).sum() == len(pts_ids), "some pts assigned to camera are not visible"
        if  num_visible_pts != num_pts:
            num_invisible_pts = num_pts - num_visible_pts
            logging.info("{} pts assigned to camera {} are not visible".format(num_invisible_pts, camera_ids[i]))

    return (focus_distance, loss, assignment)

def optimization_step_group_views(pts, pts_normals, max_pts_set_per_camera, pts_set_per_camera,\
                                    camera_positions, camera_dirs, camera_ids_group,\
                                        losses_per_camera_indiv, camera_params, loss_params):
    """
        Solve the focus distance for each camera given the associated point set, using the group views optimization.

    Args:
        pts: point set 1, np array of shape (num points x 3)
        pts_normals: point normals, np array of shape (num points x 3)
        max_pts_set_per_camera: list of set of points for each camera (in point ids, 1-d np array)
        pts_set_per_camera: list of set of points for each camera (in point ids, 1-d np array)
        camera_positions: list of camera positions, np array of shape (num cameras x 3)
        camera_dirs: list of camera directions, np array of shape (num cameras x 3)
        camera_ids_group: list of camera pairs (in camera ids, 1-d np array)
        losses_per_camera_indiv: list of individual losses for each camera pair
        camera_params
        loss_params
    Returns:
        total_loss:
        focus_distances:  list of focus distances
        pts_in_dof_per_view: list of set of points in DOF for each camera (in point ids, 1-d np array)
        pts_out_of_dof_all: list of points out of DOF for all cameras (in point ids)
        num_pts_in_focus: the total number of points in focus
    """

    losses_camera_group_indiv = []
    losses_camera_group_common = []
    focus_distances_camera_group = []
    assignment_camera_group = []

    # iterate over the combinations of group views, and solve for focus distance for each combination
    for camera_ids in camera_ids_group:
        logging.info("camera_ids_group: {}".format(camera_ids))

        # merge point set
        pts_set_ids = merge_pts_set(pts_set_per_camera, camera_ids)
        points = pts[pts_set_ids]
        normals = pts_normals[pts_set_ids]
        max_pts_set_per_camera_group = [max_pts_set_per_camera[camera_id] for camera_id in camera_ids]
        vis_pts_set_per_camera_group = []
        for vis_pts_ids in max_pts_set_per_camera_group:
            vis_pts_ids = np.arange(len(pts_set_ids))[np.isin(pts_set_ids, vis_pts_ids)]
            vis_pts_set_per_camera_group.append(vis_pts_ids) # update vis_ids in pts)
        camera_positions_group = [camera_positions[camera_id] for camera_id in camera_ids]
        camera_dirs_group = [camera_dirs[camera_id] for camera_id in camera_ids]

        focus_distance_group, loss_group, assignment = optimize_group_views(camera_ids, pts_set_ids, points, normals, vis_pts_set_per_camera_group,\
                                                        camera_positions_group, camera_dirs_group, camera_params, loss_params)

        losses_indiv = losses_per_camera_indiv[np.array(camera_ids).astype(int)] # (num_group_cameras,)
        loss_indiv = losses_indiv.sum()
        losses_camera_group_indiv.append(loss_indiv)
        losses_camera_group_common.append(loss_group)
        focus_distances_camera_group.append(focus_distance_group)
        assignment_camera_group.append(assignment)

        # sanity check: KViews should only decrease the loss
        diff_threshold = 0.001
        loss_diff = loss_group - loss_indiv
        if loss_diff > diff_threshold and loss_group != float('inf'):
            logging.info("[ERROR]: optimizing individual camera has smaller loss (diff: {})".format(loss_diff))

    # wrap the results
    result = wrap_em_common_result(losses_camera_group_indiv, losses_camera_group_common,\
                focus_distances_camera_group, assignment_camera_group)

    return result

def optimize_group_views_worker(camera_ids, pts_set_ids, pts, pts_normals, vis_pts_set_per_camera,\
                        camera_positions_group, camera_dirs_group, camera_params, loss_params,\
                        losses_camera_group_common_dict, focus_distances_camera_group_dict, assignment_camera_group_dict, debug=False):
    """
        Optimize the focus distances for a group of cameras:
        - exclude points that are only visible to one camera (the assignment for these points is trivial)
        - solve the foucs optimization problem

        Args:
            camera_ids:
            pts_set_ids: in raw pts id
            pts: 
            pts_normals:
            vis_pts_set_per_camera: in range 0 - len(pts)
            camera_positions_group:
            camera_dirs_group:
            camera_params
            loss_params
            losses_camera_group_common_dict
            focus_distances_camera_group_dict
            assignment_camera_group_dict
    """

    # initialize
    focus_distance = [0] * len(camera_positions_group)
    loss = float('inf')
    assignment = np.ones(len(pts)) * -1

    if debug:
        # logging.debug("Num of pts: {}".format(len(pts)))
        print("Num of pts: {}".format(len(pts)))

    # trivial case 1:
    if len(pts_set_ids) == 0: # no points assigned to this camera group
        # logging.debug("[Trivial Case]: no points assigned to this camera group.")
        if debug:
            print("[Trivial Case]: no points assigned to this camera group.")
        losses_camera_group_common_dict[camera_ids] = loss
        focus_distances_camera_group_dict[camera_ids] = focus_distance
        assignment_camera_group_dict[camera_ids] = assignment

    # solve: NOTE: assignment is in the order of (0 - number of cameras - 1)
    focus_distance, loss, assignment = Solver.solveFocusDistanceMultiViewsAdaptive(pts, pts_normals, vis_pts_set_per_camera,\
                                        camera_positions_group, camera_dirs_group, camera_params, loss_params)

    # Sanity check: see if points assigned to the cameras are visible (in the maximum set of points)
    for i, vis_pts_set in enumerate(vis_pts_set_per_camera):
        pts_ids = np.where(assignment==i)[0] # in range 0-len(pts)
        num_pts = len(pts_ids)
        num_visible_pts = np.in1d(pts_ids, vis_pts_set).sum()
        if debug:
            # logging.debug("{} pts assigned to camera {}.".format(num_visible_pts, camera_ids[i]))
            print("{} pts assigned to camera {}.".format(num_visible_pts, camera_ids[i]))
        # assert np.in1d(pts_ids, max_pts_set).sum() == len(pts_ids), "some pts assigned to camera are not visible"
        if  num_visible_pts != num_pts:
            num_invisible_pts = num_pts - num_visible_pts
            if debug:
                # logging.info("{} pts assigned to camera {} are not visible".format(num_invisible_pts, camera_ids[i]))
                print("{} pts assigned to camera {} are not visible".format(num_invisible_pts, camera_ids[i]))

    losses_camera_group_common_dict[camera_ids] = loss
    focus_distances_camera_group_dict[camera_ids] = focus_distance
    assignment_camera_group_dict[camera_ids] = assignment

    return 

def optimization_step_group_views_parallel(pts, pts_normals, max_pts_set_per_camera, pts_set_per_camera,\
                                    camera_positions, camera_dirs, camera_ids_group,\
                                        losses_per_camera_indiv, camera_params, loss_params, debug=False):

    """
        Solve the focus distance for each camera given the associated point set.

    Args:
        pts: point set 1, np array of shape (num points x 3)
        pts_normals: point normals, np array of shape (num points x 3)
        max_pts_set_per_camera: list of set of points for each camera (in point ids, 1-d np array)
        pts_set_per_camera: list of set of points for each camera (in point ids, 1-d np array)
        camera_position: list of camera positions, np array of shape (num cameras x 3)
        camera_dirs: list of camera directions, np array of shape (num cameras x 3)
        camera_ids_group: list of camera pairs (in camera ids, 1-d np array)
        losses_indiv: list of individual losses for each camera pair
    Returns:
        total_loss:
        focus_distances:  list of focus distances
        pts_in_dof_per_view: list of set of points in DOF for each camera (in point ids, 1-d np array)
        pts_out_of_dof_all: list of points out of DOF for all cameras (in point ids)
        num_pts_in_focus: the total number of points in focus
    """
    losses_camera_group_indiv = []
    losses_camera_group_common = []
    focus_distances_camera_group = []
    assignment_camera_group = []

    # create a manager to share data between processes
    manager = Manager()
    losses_camera_group_common_dict = manager.dict()
    focus_distances_camera_group_dict = manager.dict()
    assignment_camera_group_dict = manager.dict()

    cpu_count = os.cpu_count()
    pool = Pool(processes=cpu_count)

    # iterate over the combinations of group views, and solve for focus distance for each combination
    logging.info("Start parallel processing with {} processes.".format(cpu_count))
    logging.info("num of camera pairs: {}".format(len(camera_ids_group)))
    t0 = time.perf_counter()
    for camera_ids in camera_ids_group:
        # logging.info("camera_ids_group: {}".format(camera_ids))

        # merge point set
        pts_set_ids = merge_pts_set(pts_set_per_camera, camera_ids)
        points = pts[pts_set_ids]
        normals = pts_normals[pts_set_ids]
        max_pts_set_per_camera_group = [max_pts_set_per_camera[camera_id] for camera_id in camera_ids]
        vis_pts_set_per_camera_group = []
        for vis_pts_ids in max_pts_set_per_camera_group:
            vis_pts_ids = np.arange(len(pts_set_ids))[np.isin(pts_set_ids, vis_pts_ids)]
            vis_pts_set_per_camera_group.append(vis_pts_ids) # update vis_ids in pts)
        camera_positions_group = [camera_positions[camera_id] for camera_id in camera_ids]
        camera_dirs_group = [camera_dirs[camera_id] for camera_id in camera_ids]

        # solve
        pool.apply_async(optimize_group_views_worker, args=(camera_ids, pts_set_ids, points, normals, vis_pts_set_per_camera_group,\
                        camera_positions_group, camera_dirs_group, camera_params, loss_params,\
                        losses_camera_group_common_dict, focus_distances_camera_group_dict, assignment_camera_group_dict, debug))
    pool.close()
    pool.join()
    t1 = time.perf_counter()
    logging.info("Pool join takes {:.3f} s".format(t1-t0))

    # rearrange the results
    for camera_ids in camera_ids_group:
        # compare individual and common maximization
        losses_indiv = losses_per_camera_indiv[np.array(camera_ids).astype(int)] # (num_group_cameras,)
        loss_indiv = losses_indiv.sum()

        loss_group = losses_camera_group_common_dict[camera_ids]
        losses_camera_group_indiv.append(loss_indiv)
        losses_camera_group_common.append(loss_group)
        focus_distances_camera_group.append(focus_distances_camera_group_dict[camera_ids])
        assignment_camera_group.append(assignment_camera_group_dict[camera_ids])


        # sanity check: KViews should only decrease the loss
        diff_threshold = 0.001
        loss_diff = loss_group - loss_indiv
        if loss_diff > diff_threshold and loss_group != float('inf'):
            logging.info("[ERROR]: optimizing individual camera has smaller loss (diff: {})".format(loss_diff))

    # wrap the results
    result = wrap_em_common_result(losses_camera_group_indiv, losses_camera_group_common,\
                focus_distances_camera_group, assignment_camera_group)

    return result

def greedyUpdate(camera_ids_group, focus_distances, pts_set_per_camera,\
                    losses_camera_group_indiv, losses_camera_group_common,\
                        focus_distances_camera_group, assignment_camera_group, debug=False):
    """
        Update focus distance from common view optimization in a greedy algorithm.

    Args:
        camera_ids_group: 
        focus_distances:
        pts_set_per_camera: list of set of points for each camera (in point ids, 1-d np array)
        losses_camera_group_indiv: list of individual losses for each camera pair
        losses_camera_group_common: list of common losses for each camera pair
        focus_distances_camera_group: list of focus distances for each camera pair
        assignment_camera_group: list of assignment for each camera pair

    Returns:
        focus_distances_updated:
        pts_set_per_camera_updated:
    """

    # Greedy update for focus distance:
    # 1. update the focus distance for the pair of cameras with the largest loss reduction
    # 2. remove the pair of cameras that include cameras in (1) from the list
    # 3. repeat (1) and (2) until no more pairs of cameras
    focus_distances_updated = focus_distances.copy()
    pts_set_per_camera_updated = pts_set_per_camera.copy()

    loss_threshold = -0.01
    losses_update = losses_camera_group_common - losses_camera_group_indiv
    while losses_update.min() < loss_threshold:
        list_id = np.argmin(losses_update)
        loss_decrease = -losses_update[list_id]
        camera_ids = camera_ids_group[list_id]

        if debug:
            print("updating with camera ids {}, loss decrease: {}".format(camera_ids, loss_decrease))

        # merge point set
        pts_set_ids = merge_pts_set(pts_set_per_camera, camera_ids)

        # update
        focus_distance = focus_distances_camera_group[list_id]
        assignment = assignment_camera_group[list_id]
        for i, camera_id in enumerate(camera_ids):
            focus_distances_updated[camera_id] = focus_distance[i]
            pts_set_camera_updated = pts_set_ids[assignment == i]
            pts_set_per_camera_updated[camera_id] = pts_set_camera_updated
        
        # Set the loss for the camera group including the updated cameras to inf to avoid updating them again
        for i, camera_ids_temp in enumerate(camera_ids_group):
            for camera_id in camera_ids:
                if camera_id in camera_ids_temp:
                    losses_update[i] = float('inf')
                    break

    return focus_distances_updated, pts_set_per_camera_updated

def main():

    import argparse
    import glob
    import shutil

    # argument set-up
    parser = argparse.ArgumentParser(description="Run k-view algorithm for focus distance optimization.")
    parser.add_argument("-i", "--input", type=str, help="Path to subject folder")
    parser.add_argument("-em", "--em", type=str, default="output_em", help="Em folder")
    parser.add_argument("-iter", "--iter", type=int, default=-1, help="Iteration in EM to use as the initialization for KViews, default: -1  (last)")
    parser.add_argument("-p", "--params", type=str, help="Path to params file")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode (store true)")
    parser.add_argument("-v", "--verbosity", type=str, default="info", help="Verbosity (default: info)")
    parser.add_argument("-s", "--save_dir", type=str, default="output_kview", help="Path to save dir")

    # # Parse the command line arguments to an object
    args = parser.parse_args()
    if not args.input:
        print("No input data is provided.")
        print("For help type --help")
        exit()

    root_dir = os.path.join(args.input, args.em)
    param_file = args.params
    DEBUG = args.debug
    SAVE_DIR = os.path.join(args.input, args.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)
    setup_logging(args.verbosity, SAVE_DIR)
    log_args(args)

    # load params
    params = load_params(param_file)
    NUM_K_VIEWS = params['alg']['num_k_views']
    RING = params['alg']['ring']
    NUM_ITERS = params['alg']["num_k_views_iters"]
    camera_params = get_camera_params(params) 
    loss_params = get_loss_params(params)
    log_args(params)

    # load output data for initizialiation
    outfnames = glob.glob(os.path.join(root_dir, "*.pkl"))
    # remove irrelevant files 
    for input_file in outfnames.copy():
        basename = os.path.basename(input_file)
        if "output_iter" not in basename:
            outfnames.remove(input_file)
    outfnames.sort()
    outfname = outfnames[args.iter]
    cameras_network, focus_distances, max_pts_set_per_camera, pts, pts_normals,\
                pts_set_per_camera = load_output(outfname)
    pts_set_per_camera = np.array(pts_set_per_camera, dtype='object') # debug
    shutil.copy(param_file, os.path.join(SAVE_DIR, "params.yml")) # copy params to the input folder

    logging.info("Applying K-View optimization to {}".format(outfname))

    # debug: check pts assignment result
    logging.debug("num of pts: ".format(len(pts)))
    for camera_id, pts_set in enumerate(pts_set_per_camera):
        if len(pts_set) == 0:
            logging.debug("camera {} does not have assigned points.".format(camera_id))

    # point set 2: camera poses
    camera_positions, camera_dirs, _ = cameras_network.parseCameras()
    if NUM_K_VIEWS == 2:
        camera_ids_group = cameras_network.createCameraPairsID(ring=RING)
    elif NUM_K_VIEWS == 3:
        camera_ids_group = cameras_network.createCameraTripletID(ring=RING)
    else:
        raise NotImplementedError("NUM_K_VIEWS > 3 is not implemented yet.")
    if DEBUG:
        print(camera_ids_group)

    ### from initialization, ex: EM
    losses_per_camera_indiv = analyze_loss_per_camera(pts, pts_normals, pts_set_per_camera, max_pts_set_per_camera,\
                                                    camera_positions, camera_dirs, focus_distances,\
                                                    camera_params=camera_params, loss_params=loss_params)
    loss_initial = sum(losses_per_camera_indiv)
    logging.info("Initial total loss: {}".format(loss_initial))

    total_loss = loss_initial
    t00 = time.perf_counter()
    for i in range(NUM_ITERS):
        t0 = time.perf_counter()
        logging.info("Iteration {}".format(i))

        if i == 0: # initialization
            focus_distances_updated = focus_distances.copy()
            pts_set_per_camera_updated = pts_set_per_camera.copy()
        else:
            # update
            # result = optimization_step_group_views(pts, pts_normals, max_pts_set_per_camera, pts_set_per_camera,\
            #     camera_positions, camera_dirs, camera_ids_group, losses_per_camera_indiv, camera_params, loss_params)
            result = optimization_step_group_views_parallel(pts, pts_normals, max_pts_set_per_camera, pts_set_per_camera,\
                camera_positions, camera_dirs, camera_ids_group, losses_per_camera_indiv, camera_params, loss_params, debug=DEBUG)
            t1 = time.perf_counter()
            logging.info("K views otimization takes {:.3f} s".format(t1-t0))

            pickle_file = outfname.replace(".pkl", "_{}_views_{}_iter_{}.pkl".format(NUM_K_VIEWS, RING, i))
            pickle_file = pickle_file.replace("output", "result")
            pickle_file = os.path.join(SAVE_DIR, os.path.basename(pickle_file))
            # save the results
            with open(pickle_file, "wb") as f:
                logging.info("Saving results to {}".format(pickle_file))
                pickle.dump(result, f)

            losses_camera_group_indiv, losses_camera_group_common,\
                focus_distances_camera_group, assignment_camera_group = get_em_common_result(result)

            t0 = time.perf_counter()
            # Greedy update
            focus_distances_updated, pts_set_per_camera_updated = greedyUpdate(camera_ids_group, focus_distances, pts_set_per_camera,\
                                            losses_camera_group_indiv, losses_camera_group_common, focus_distances_camera_group, assignment_camera_group, debug=DEBUG)
            t1 = time.perf_counter()
            logging.info("greedy update takes {:.3f} s".format(t1-t0))

            losses_per_camera_common = analyze_loss_per_camera(pts, pts_normals, pts_set_per_camera_updated, max_pts_set_per_camera,\
                                                            camera_positions, camera_dirs, focus_distances_updated,\
                                                            camera_params=camera_params, loss_params=loss_params)
            loss_common = sum(losses_per_camera_common)
            logging.info("Total loss after k view: {}".format(loss_common))

            # termination condition
            if loss_common >= total_loss:
                break

            # update for next iteration
            focus_distances = focus_distances_updated.copy()
            pts_set_per_camera = pts_set_per_camera_updated.copy()
            losses_per_camera_indiv = analyze_loss_per_camera(pts, pts_normals, pts_set_per_camera_updated, max_pts_set_per_camera,\
                                                            camera_positions, camera_dirs, focus_distances_updated,\
                                                            camera_params=camera_params, loss_params=loss_params)
            total_loss = loss_common

        pickle_file = os.path.basename(outfname).replace(".pkl", "_{}_views_{}_iter_{}.pkl".format(NUM_K_VIEWS, RING, i))
        save_output(cameras_network, focus_distances_updated, max_pts_set_per_camera, pts, pts_normals,\
                    pts_set_per_camera_updated, outfname=pickle_file, save_dir=SAVE_DIR)

    t11 = time.perf_counter()
    duration = t11 - t00
    if i == 0: # no iteration
        logging.info("Total Time: {}sec; {} sec per iteration.".format(duration, duration))
    else:
        logging.info("Total Time: {}sec; {} sec per iteration.".format(duration, duration/(i))) # exclude the first iteration

if __name__ == "__main__":
    main()