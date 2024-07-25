import trimesh
import numpy as np

from Camera import dof_cal

def getOffOpticalAxisLoss(pts, camera_position, camera_dir, loss_type=None, threshold_distance=600):
    """
        Get off-optica-axis loss for each point in pts given the camera position and its position.
        NOTE: This term is orthogonal to the camera viewing direction

        loss =  min(1, ||pi_{n_c}(p - p_c)|| / threshold_distance)

        p: point position
        p_c: camera position
        pi_{n_c}(): the projection function of the vector on the plane orthogonal to the camera viewing direction

    Args:
        pts: np array of shape (num points x 3)
        camera_position: np array of shape (3,)
        camera_dir: np array of shape (3,)
        loss_type: 'l0' or 'l1', 'l0' is binary, 'l1' is normalized to (0, 1)
        threshold_distance: distance threshold for the loss, if the distance is larger than the threshold, the loss is 1
    Returns:
        loss: off-optical-axis loss, np array of length (num points)
    """
    vec_cam_to_pts = pts - camera_position
    vec_cam_to_pts_ortho = vec_cam_to_pts - np.matmul(vec_cam_to_pts, camera_dir.reshape((3, 1))) * camera_dir.reshape((1, 3))
    dist = np.linalg.norm(vec_cam_to_pts_ortho, axis=1)

    pts_out_of_threshold = dist > threshold_distance
    loss = np.zeros(np.shape(dist))
    if loss_type == 'l0':
        loss[pts_out_of_threshold] = 1
    elif loss_type == 'l1':
        loss[pts_out_of_threshold] = 1
        pts_in_threshold = ~pts_out_of_threshold
        loss[pts_in_threshold] = dist[pts_in_threshold] / threshold_distance # normalize to (0, 1)

    # sanity check if the loss is normalized to (0, 1)
    if (loss < 0).any() or (loss > 1).any():
        print("Off-optical-axis loss out of range")

    return loss

def getProjectionAreaLoss(pts, pts_normals, camera_position, camera_dir, loss_type=None, threshold_area=0.000001):
    """
        Get loss of the projection area for each point in pts given the camera position and its position:
        A = cos(theta) / d^2, theta is the angle between the camera direction and the negative point normal, d is the depth of the point
    Returns:
        loss: projection area loss, np array of length (num points)
    """
    depths = trimesh.points.point_plane_distance(pts, camera_dir.flatten(), camera_position.flatten())
    incidence = np.matmul(-pts_normals, camera_dir.reshape((3, 1))).flatten() # cos(theta), -1 ~ 1
    projection_area = incidence / (depths**2)

    pts_out_of_threshold = (projection_area < threshold_area)

    loss = np.zeros(np.shape(projection_area))
    if loss_type == 'l0':
        loss[pts_out_of_threshold] = 1
    elif loss_type == 'l1':
        loss[pts_out_of_threshold] = 1
        pts_in_threshold = ~pts_out_of_threshold
        loss[pts_in_threshold] = threshold_area / projection_area[pts_in_threshold] # normalize to (0, 1)

    # sanity check if the loss is normalized to (0, 1)
    if (loss < 0).any() or (loss > 1).any():
        print("Projection area loss out of range")

    return loss
    
def getFocalLoss(pts, camera_position, camera_dir, focus_distance, camera_params, loss_type=None, num_focus_intervals=4):
    """
        Get focal loss for each point in pts given the camera position, camera viewing direction, and its focus distance.
        NOTE: L0 loss is a binary function when num_focus_intervals=4: 0.5 if the point is within camera's DOF, 1 otherwise.
        NOTE: L1 loss is a piecewise linear function,
              For a given point, 3 cases are considered: 1. out of focus, 2. in focus and in front of the focal plane, 3. in focus and behind the focal plane
    Returns:
        loss: focal loss, np array of length (num points)
    """

    depths = trimesh.points.point_plane_distance(pts, camera_dir.flatten(), camera_position.flatten())

    # case when focus distance is 0 => assuming all the points are out of focus
    if focus_distance == 0:
        # print("WARNING: focus distance is 0, assuming all the points are out of focus, no points should be assigned to this camera")
        loss = np.ones(np.shape(depths))
        return loss

    focal_length = camera_params['focal_length']
    hyperfocal_distance = camera_params['hyperfocal_distance']
    dof_n, dof_f = dof_cal(focus_distance, focal_length, hyperfocal_distance)
    if loss_type == "l0":
        if num_focus_intervals == 4: # binary function, 0 if the point is within camera's DOF, 1 otherwise
            loss = np.ones(np.shape(depths))
            # pts within dof
            pts_in_dof = (depths > dof_n) & (depths < dof_f)
            loss[pts_in_dof] = 0

            # # NOTE: Use a tighter approximation to piecewise linear function
            # dof_n += 0.5 * (focus_distance - dof_n)
            # dof_f -= 0.5 * (dof_f - focus_distance)
        else:
            raise NotImplementedError

    elif loss_type == "l1":
        loss = np.ones(np.shape(depths)) 

        # pts within dof
        pts_front = (depths > dof_n) & (depths < focus_distance) # pts btween dof_n and focus_distance
        pts_back = (depths > focus_distance) & (depths < dof_f) # pts btween focus_distance and dof_f
        loss[pts_front] = (focus_distance - depths[pts_front]) / (focus_distance - dof_n) 
        loss[pts_back] = (depths[pts_back] - focus_distance) / (dof_f - focus_distance) 

    elif loss_type == "l2":
        raise NotImplementedError

    return loss

def getViewQualityLoss(pts, pts_normals, vis_pts_set, camera_position, camera_dir, focus_distance, camera_params, loss_params):
    """
        Get the loss of view quality for each point in pts for the camera.
        Projection area loss: depth to the camera, or the depth to the focus plane of the camera, normlalized by the largest depth value
        Off optical axis loss: normalized dot product between the normal vector of the point and the camera direction
        Focal loss:

    Args:
        pts: , np array of shape (num points x 3)
        pts_normals: normal vector associated with each point in pts, np array of shape (num points x 3)
        vis_pts_set: ids of visible points (id in pts) to the camera
        camera_position: list of camera positions, np array of shape (num cameras x 3)
        camera_dir: list of camera directions, np array of shape (num cameras x 3)
        focus_distance: list of focus distances, np array of shape (num cameras, )
        camera_params
        loss_params
    Returns:
        loss: np array of shape (num points)
    """
    mode = loss_params['mode']
    weights =  loss_params['weights']
    focal_loss_type = loss_params['focal_loss_type']
    projection_area_loss_type = loss_params['projection_area_loss_type']
    off_optical_axis_loss_type = loss_params['off_optical_axis_loss_type']
    projection_area_threshold = loss_params['projection_area_threshold']
    off_optical_axis_threshold_distance = loss_params['off_optical_axis_threshold_distance']
    num_focus_intervals = loss_params['num_focus_intervals']

    # reshape
    pts = pts.reshape((-1, 3))
    pts_normals = pts_normals.reshape((-1, 3))

    # focal: depth to the focal plane # non-negative: bounded by the size of the cylinder trajectory
    focal_loss = getFocalLoss(pts, camera_position, camera_dir, focus_distance,\
                            camera_params=camera_params, loss_type=focal_loss_type, num_focus_intervals=num_focus_intervals)

    # projection area
    projection_area_loss = getProjectionAreaLoss(pts, pts_normals, camera_position, camera_dir,\
                            loss_type=projection_area_loss_type, threshold_area=projection_area_threshold)

    # off optical axis
    off_optical_axis_loss = getOffOpticalAxisLoss(pts, camera_position, camera_dir,\
                            loss_type=off_optical_axis_loss_type, threshold_distance=off_optical_axis_threshold_distance)

    if mode == "add":
        loss = weights[0] * projection_area_loss + weights[1] * focal_loss + weights[2] *  off_optical_axis_loss
    elif mode == "mul":
        loss = weights[0] * np.exp(projection_area_loss) + weights[1] * np.exp(focal_loss) + weights[2] * np.exp(off_optical_axis_loss) 
    else:
        raise NotImplementedError

    # update loss accounting for visibility
    invisible_pts_ids = np.setdiff1d(np.arange(len(pts)), vis_pts_set)
    loss[invisible_pts_ids] = 1

    return loss

def getViewQualiyLossAllCameras(pts, pts_normals, vis_pts_set_per_camera, camera_positions, camera_dirs, focus_distances, camera_params, loss_params):
    """
        Get the loss of view quality for each point in pts for each camera.

        The score is a weighted sum of the distance and alignment scores.
        Distance score: depth to the camera, or the depth to the focus plane of the camera, normlalized by the largest depth value
        Alignment score: normalized dot product between the normal vector of the point and the camera direction

    Args:
        pts: point set 1, np array of shape (num points x 3)
        pts_normals: normal vector associated with each point in pts, np array of shape (num points x 3)
        pts_vis_list:
        camera_position: list of camera positions, np array of shape (num cameras x 3)
        camera_dirs: list of camera directions, np array of shape (num cameras x 3)
        focus_distances: list of focus distances, np array of shape (num cameras, )
        weights: weights for combining dist and alignment scores (dist_weight, alignment_weight)
    Returns:
        loss: np array of shape (num points x num cameras)
    """
    loss = [] # num points x num cameras

    camera_positions = np.array(camera_positions).reshape((-1, 3))
    camera_dirs = np.array(camera_dirs).reshape((-1, 3))
    focus_distances = np.array(focus_distances).reshape((-1, 1))

    for vis_pts_set, camera_position, camera_dir, focus_distance in zip(vis_pts_set_per_camera, camera_positions, camera_dirs, focus_distances):
        loss_single_camera = getViewQualityLoss(pts, pts_normals, vis_pts_set, camera_position, camera_dir, focus_distance, camera_params, loss_params)
        loss.append(loss_single_camera)
    loss = np.array(loss)
    loss = loss.T

    return loss 