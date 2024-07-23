import numpy as np
from Loss import getViewQualityLoss

def analyze_loss_per_camera(pts, pts_normals, pts_set_per_camera, vis_pts_set_per_camera, camera_positions, camera_dirs, focus_distances, camera_params, loss_params):
    """
        Analyze the loss per camera.
        Return:
            loss_per_camera: loss per camera, shape (num_cameras,)
    """
    loss_per_camera = []
    for pts_set_ids, vis_pts_ids, camera_position, camera_dir, focus_distance in zip(pts_set_per_camera, vis_pts_set_per_camera,\
                                                                                camera_positions, camera_dirs, focus_distances):
        if len(pts_set_ids) == 0:
            loss_per_camera.append(0) # assign to zero loss if there are no points for the camaera
            continue

        pts_set_ids = np.array(pts_set_ids)
        vis_pts_ids = np.array(vis_pts_ids)
        vis_ids = np.arange(len(pts_set_ids))[np.isin(pts_set_ids, vis_pts_ids)] # update vis_ids in pts
        points = pts[pts_set_ids]
        normals = pts_normals[pts_set_ids]

        loss = getViewQualityLoss(points, normals, vis_ids, camera_position, camera_dir, focus_distance, camera_params=camera_params, loss_params=loss_params)
        loss_per_camera.append(np.sum(loss))

    loss_per_camera = np.array(loss_per_camera)
    return loss_per_camera

def analyze_loss_per_point(pts, pts_normals, pts_set_per_camera, vis_pts_set_per_camera, camera_positions, camera_dirs, focus_distances, camera_params, loss_params):
    """
        Analyze the loss per point.
        Return:
            loss_per_point: loss per point, shape (num_pts,)
        NOTE: For a point that is not visible in any camera, the loss is zero here.
    """
    loss_per_point = np.zeros(pts.shape[0])
    for pts_set_ids, vis_pts_ids, camera_position, camera_dir, focus_distance in zip(pts_set_per_camera, vis_pts_set_per_camera,\
                                                                                camera_positions, camera_dirs, focus_distances):
        if len(pts_set_ids) == 0:
            continue
        pts_set_ids = np.array(pts_set_ids)
        vis_pts_ids = np.array(vis_pts_ids)
        vis_ids = np.arange(len(pts_set_ids))[np.isin(pts_set_ids, vis_pts_ids)] # update vis_ids in pts
        points = pts[pts_set_ids]
        normals = pts_normals[pts_set_ids]
        loss = getViewQualityLoss(points, normals, vis_ids, camera_position, camera_dir, focus_distance, camera_params=camera_params, loss_params=loss_params)
        loss_per_point[pts_set_ids] = loss

    return loss_per_point