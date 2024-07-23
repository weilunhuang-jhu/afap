import numpy as np
import itertools
import trimesh
import time
import logging

from Camera import focus_interval_cal 
from Loss import getViewQualityLoss, getViewQualiyLossAllCameras

RESOLUTION = 1 # mm
NUM_BINS = 32

def getFocusIntervals(depths, camera_params, num_focus_intervals=4):
    """
        Given a set of depths, derive the focus intervals for the camera.
        This is done by:
            1. Calculate the near and far limit distance so that a point at given depth can be covered in focus,
            2. Divide the interval between:
                (a) 0 to the near limit
                (b) near limit to depth: (num_focus_intervals - 2) // 2
                (c) depth to far limit: (num_focus_intervals - 2) // 2
                (d) far limit to infinity into num_focus_intervals intervals - 2,
            3. Get the breakpoints of the intervals, neglecting 0, depth, and infinity
            4. Get the change of cost at each breakpoint
               NOTE: the cost is decreasing from 1 to 0 from near limit to depth, and increasing from 0 to 1 from depth to far limit
               NOTE: The change of cost is 1/
               NOTE: The change of cost is 0 at depth
            5. Construct the focus intervals by
                (a) associate the breakpoints with the change of cost
                (b) sort the breakpoints

        Args:
            depths: depths of the points, np array of shape (num points, )
            camera_params: camera parameters
            num_focus_intervals: number of focus intervals
        Returns:
            t: focus intervals, np array of shape (num_focus_intervals - 2, 2)
            NOTE: the focus intervals are represented by (breakpoint, change of cost), sorted by the breakpoint
    """
    focal_length = camera_params['focal_length']
    hyperfocal_distance = camera_params['hyperfocal_distance']
    s_0, s_1 = focus_interval_cal(depths, focal_length, hyperfocal_distance)
    if num_focus_intervals == 4: # special case: use the near and far limit with cost 1 and 0
        t_0 = np.column_stack((s_0, -np.ones(s_0.shape)))
        t_at_depths = np.column_stack((depths, np.zeros(depths.shape)))
        t_1 = np.column_stack((s_1, np.ones(s_1.shape)))

    else:
        num_breakpoint_per_side = num_focus_intervals // 2 # not considering 0 and infinity
        change_of_cost = 1/num_breakpoint_per_side

        # construct the breakpoints of intervals: s_0 to depths, cost is decreasing from 1 to 0
        breakpoints_0 = np.linspace(s_0, depths, num_breakpoint_per_side).T # shape of (num_pts, num_breakpoint_per_side)
        breakpoints_0 = breakpoints_0[:, :-1] # exclude the last one (depths), considered in the next part
        breakpoints_0 = breakpoints_0.flatten() # shape of (num_pts * num_breakpoint_per_side - 1, )
        t_0 = np.column_stack((breakpoints_0, -change_of_cost * np.ones(breakpoints_0.shape)))

        t_at_depths = np.column_stack((depths, np.zeros(depths.shape)))

        # construct the breakpoints of intervals: depths to s_1, cost is increasing from 0 to 1
        breakpoints_1 = np.linspace(depths, s_1, num_breakpoint_per_side).T # shape of (num_pts, num_breakpoint_per_side)
        breakpoints_1 = breakpoints_1[:, 1:] # exclude the first one (depths)
        breakpoints_1 = breakpoints_1.flatten() # shape of (num_pts * num_breakpoint_per_side - 1, )
        t_1 = np.column_stack((breakpoints_1, change_of_cost * np.ones(breakpoints_1.shape)))

    t = np.vstack((t_0, t_at_depths, t_1))
    t = t[np.argsort(t[:,0], axis=0)]

    return t

def solveFocusDistanceByWeightedCount(depths, camera_params, num_focus_intervals=4, debug=False):
    """
        Given a set of depths, derive the optimal focus distance that minimize the cost.
    """
    focus_intervals = getFocusIntervals(depths, camera_params, num_focus_intervals=num_focus_intervals)
    cost = len(depths) + np.cumsum(focus_intervals[:,1])
    id_opt = np.argmin(cost)
    s_opt = (focus_intervals[id_opt, 0] + focus_intervals[id_opt + 1, 0]) / 2 # use the middle point of the interval

    return s_opt

def solveFocusDistanceGrid(pts, pts_normals, vis_pts_set,\
                            camera_position, camera_dir, camera_params, loss_params,\
                                  focus_distance_prev=None, num_bins=NUM_BINS):
    """
        Solve the optimal focus distance
        NOTE: the solution is dependent on the number of bins (sampling density)
        NOTE: Need to include the previous focus distance in the grid for correctness (monotonically decreasing the loss)
    """
    assert len(pts) > 0, "[ERROR] No point assigned to the group inside solveFocusDistanceUniformlySampling"

    pts = pts.reshape((-1, 3))
    depths = trimesh.points.point_plane_distance(pts, camera_dir.flatten(), camera_position.flatten())

    min_loss = float('inf')
    focus_distance_opt = 0

    # focus_distances = np.arange(np.min(depths), np.max(depths), resolution)
    # construct a list of focus distances spanning from min depth to max depth, given the number of bins
    focus_distances = np.linspace(np.min(depths), np.max(depths), num_bins)

    # add the previous focus distance to the list if it is not included, make sure the loss is monotonically decreasing
    if focus_distance_prev is not None and focus_distance_prev not in focus_distances:
        focus_distances = np.append(focus_distances, focus_distance_prev)

    for focus_distance in focus_distances:
        loss = getViewQualityLoss(pts, pts_normals, vis_pts_set, camera_position, camera_dir, focus_distance, camera_params, loss_params)
        loss = np.sum(loss)
        if loss < min_loss:
            min_loss = loss
            focus_distance_opt = focus_distance

    return focus_distance_opt, min_loss

def solveFocusDistanceAdaptive(pts, pts_normals, vis_pts_set, camera_position, camera_dir,\
                                camera_params, loss_params, debug=False):
    """
        Given a camera and a set of points, derive a focus distance that minimize the view quality cost.

        For a single camera, if the focal loss is binary, finding the optimal focus distance can be reduced to
        finding the focus distance at which the depth of field cover the most number of points.

        If the focal loss is piecewise constant, 

        If the focal loss is not binary, the focus distance is derived by minimizing the view quality loss.

        The resolution of the focus distance is adaptive to the depth of the points.

        NOTE: assume at least 1 point when calling this function
    """
    assert len(pts) > 0, "[ERROR] No point assigned to the group inside solveFocusDistanceAdaptive"

    pts = pts.reshape((-1, 3))
    depths = trimesh.points.point_plane_distance(pts, camera_dir.flatten(), camera_position.flatten())

    # trival case: only one point => set focus distance to the depth of the point and the focal loss would be 0
    if len(depths) == 1:
        focus_distance_opt = depths[0]
        min_loss = getViewQualityLoss(pts, pts_normals, vis_pts_set, camera_position, camera_dir, focus_distance_opt, camera_params, loss_params)
        min_loss = min_loss.sum()
        return focus_distance_opt, min_loss

    # NOTE:  Do we need to handle previous focus distance: check if the previous focus distance is included in any of the intervals?
    if debug:
        print("depths: ", depths)

    # TODO: calculate loss and derive from argmin
    if loss_params['focal_loss_type'] == "l0":
        num_focus_intervals = loss_params['num_focus_intervals']
        focus_distance_opt = solveFocusDistanceByWeightedCount(depths, camera_params, num_focus_intervals=num_focus_intervals, debug=debug)
    elif loss_params['focal_loss_type'] == "l1":
        raise NotImplementedError("Not implemented yet")
        # focus_distance_opt = solveFocusDistancePiecewiseLinear(depths, camera_params)

    min_loss = getViewQualityLoss(pts, pts_normals, vis_pts_set, camera_position, camera_dir, focus_distance_opt, camera_params, loss_params)
    if debug:
        print("loss: ", min_loss)
    min_loss = np.sum(min_loss)

    return focus_distance_opt, min_loss

def solveFocusDistanceMultiViewsAdaptive(pts, pts_normals, vis_pts_set_per_camera, camera_positions, camera_dirs, camera_params, loss_params):
    """
        Derive focus distances for a group of cameras to minimize the view quality loss.
        The resolution of the focus distance is adaptive to the depth of the points.
        Args:
            pts: point set 1, np array of shape (num points x 3)
            pts_normals: normal vector associated with each point in pts, np array of shape (num points x 3)
            vis_pts_set_per_camera
            camera_positions: list of camera positions, np array of shape (num cameras x 3)
            camera_dirs: list of camera directions, np array of shape (num cameras x 3)
            camera_params:
            loss_params:
        Return:
            focus_distances_opt: the focus distances that minimizes the view quality loss
            min_loss:
            assigned_view_opt: the assigned view for each point (in camera_id starting from 0 to num cameras - 1)
    """
    pts = pts.reshape((-1, 3))
    focus_distance_opt = [0] * len(camera_positions)
    assigned_view_opt = np.ones(len(pts)) * -1
    min_loss = float('inf')

    assert len(pts) > 0, "No point assigned to the group, this should not happen"

    focus_grids = []
    for camera_id, camera_position, camera_dir in zip(\
                range(len(camera_positions)), camera_positions, camera_dirs):
        depths = trimesh.points.point_plane_distance(pts, camera_dir.flatten(), camera_position.flatten())
        if len(depths) == 1: # trival case: only one point => set focus distance to the depth of the point
            focus_grid = np.array([depths[0]])
        else: # more than 1 points
            focus_intervals = getFocusIntervals(depths, camera_params)
            focus_grid = (focus_intervals[:-1,0] + focus_intervals[1:,0]) / 2
            # logging.debug("camera id: {} with {} focus grid".format(camera_id, len(focus_grid)))
        focus_grids.append(focus_grid)

    group_focus_distances = list(itertools.product(*focus_grids))
    # logging.debug("num of group focus distnace: {}".format(len(group_focus_distances)))

    # Iterate through all possible focus distances
    loss = [] # loss for each focus distance, shape of (num focus distances combinations, num pts, num cameras in a group)
    assigned_views = []

    for focus_distances in group_focus_distances:
        loss_per_pt = getViewQualiyLossAllCameras(pts, pts_normals, vis_pts_set_per_camera,\
                                                camera_positions, camera_dirs, focus_distances=focus_distances,\
                                                camera_params=camera_params, loss_params=loss_params)
        assigned_view_f = np.argmin(loss_per_pt, axis=1)
        loss_f = np.min(loss_per_pt, axis=1).sum()
        assigned_views.append(assigned_view_f)
        loss.append(loss_f)

    loss = np.array(loss)
    opt_id = np.argmin(loss)
    # get the focus distance with the smallest loss (the best focus distance)
    focus_distance_opt = group_focus_distances[opt_id]
    assigned_view_opt = assigned_views[opt_id]
    min_loss = loss[opt_id]

    return focus_distance_opt, min_loss, assigned_view_opt