import numpy as np
import cv2
from Camera import get_pts_in_dof_mask
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

def analyze_focus_per_point(pts, pts_set_per_camera,\
                                    camera_positions, camera_dirs, focus_distances, camera_params):
    """
        Analyze if each point is in-focus or not given the point assignment \phi.
        NOTE: the pts does not include invisble points, meaning all the pt in pts is visible to at least one camera
        NOTE: if visible_set_per_camera is provided for pts_set_per_camera,
        the function will calclate if a point is covered in focus in any of the camera, instead of the assigned camera
        Return:
            pts_focus_all: unique ids of points that are in-focus, ids are in the order of pts
            pts_out_focus_all: unique ids of points that are out of focus, ids are in the order of pts
    """

    focal_length = camera_params['focal_length']
    hyperfocal_distance = camera_params['hyperfocal_distance']
    pts_focus_all = []

    for pts_set_ids, camera_position, camera_dir, focus_distance in zip(pts_set_per_camera, camera_positions,\
                                                                    camera_dirs, focus_distances):
        if len(pts_set_ids) == 0:
            continue
        pts_set_ids = np.array(pts_set_ids)
        points = pts[pts_set_ids]
        # Get vertices of the mesh within FOV and DOF by calculating depth of each point and filter by depth given points within FOV
        pts_focus = get_pts_in_dof_mask(points, camera_position, camera_dir, focus_distance, focal_length=focal_length, hyperfocal_distance=hyperfocal_distance)
        pts_good_quality_focus = pts_set_ids[pts_focus]
        pts_focus_all.extend(pts_good_quality_focus)
    pts_focus_all = np.unique(pts_focus_all) # make sure there are no duplicates
    pts_out_focus_all = np.setdiff1d(np.arange(len(pts)), pts_focus_all)

    return pts_focus_all, pts_out_focus_all

def calculate_mm_per_pixel_along_vector(v, points, camera_pose, K, distortion, unit_length=10):

    H_cameraToModel = np.linalg.inv(camera_pose)

    # Create end points of the line segment along vector v
    end_point1 = points + 0.5 * unit_length * v
    end_point2 = points -  0.5 *unit_length * v
    
    # Project end points to image coordinates
    img_coords1, _ = cv2.projectPoints(end_point1.reshape((-1,1,3)), H_cameraToModel[:3,:3], H_cameraToModel[:3,3], K, distortion)
    img_coords2, _ = cv2.projectPoints(end_point2.reshape((-1,1,3)), H_cameraToModel[:3,:3], H_cameraToModel[:3,3], K, distortion)
    # squeeze the shape of img_coords, from (N,1,2) to (N,2)
    img_coords1 = img_coords1.reshape((-1,2))
    img_coords2 = img_coords2.reshape((-1,2))
    
    # Calculate distances in pixels
    # Handle case for one point
    if len(img_coords1) == 1 and len(img_coords2) == 1:
        pixel_distances = np.linalg.norm(img_coords1 - img_coords2)
    else:
        pixel_distances = np.linalg.norm(img_coords1 - img_coords2, axis=1)
    
    return unit_length / pixel_distances

def calculate_mm_per_pixel_per_camera(camera_data, points, normals, camera_pose, unit_length=10, S=30):
    # unit_length is the length of the line segment in mm
    # S is the number of vectors to sample from 0-90 degree in a circle on a plane

    # camera parameters
    # w = camera_data.img_resolution[0]
    # h = camera_data.img_resolution[1]
    K = camera_data.intrinsics.K
    distortion = camera_data.intrinsics.dist
    # print("distortion: ", distortion)

    # Uniformly sample S unit vectors from 0-90 degree in a circle on a plane
    theta = np.linspace(0, np.pi/2, S, endpoint=False)    

    # Create unit vectors on the XY plane
    v_samples = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(S)))
    
    # Normalize to ensure they are unit vectors
    v_samples = v_samples / np.linalg.norm(v_samples, axis=1)[:, np.newaxis]

    mm_per_pixel = np.zeros((len(points)))
    for v in v_samples:
        # Create two orthogonal vectors to the normal
        v1 = np.cross(normals, v)
        v1 /= np.linalg.norm(v1, axis=1)[:, np.newaxis]
        v2 = np.cross(normals, v1)
        v2 /= np.linalg.norm(v2, axis=1)[:, np.newaxis]
        
        # Calculate mm per pixel resolution along v1
        mm_per_pixel_1 = calculate_mm_per_pixel_along_vector(v1, points, camera_pose, K, distortion, unit_length)
        mm_per_pixel_2 = calculate_mm_per_pixel_along_vector(v2, points, camera_pose, K, distortion, unit_length)

        # Take the average of the two
        mm_per_pixel += (mm_per_pixel_1 + mm_per_pixel_2) / 2
    # average mm_per_pixel
    mm_per_pixel /= S
    return mm_per_pixel

def analyze_resolution_per_point(pts, pts_normals, pts_set_per_camera, max_pts_set_per_camera,\
                                    cameras, camera_poses):
    """
        Analyze resolution in mm_per_pixel for each point given the assignment \phi.
        NOTE: the pts does not include invisble points, meaning all the pt in pts is visible to at least one camera
        Return:
            resolution_per_point: the resolution for each point
    """
    pts_set_resolution = []
    for pts_set_ids, vis_pts_ids, camera_data, camera_pose in zip(pts_set_per_camera, max_pts_set_per_camera,\
                                                                    cameras, camera_poses):
        if len(pts_set_ids) == 0:
            pts_set_resolution.append(None)
            continue

        pts_set_ids = np.array(pts_set_ids)
        vis_pts_ids = np.array(vis_pts_ids)
        vis_ids = np.arange(len(pts_set_ids))[np.isin(pts_set_ids, vis_pts_ids)] # update vis_ids in pts

        points = pts[pts_set_ids]
        normals = pts_normals[pts_set_ids]
        normals = normals / np.linalg.norm(normals, axis=1).reshape((-1,1))
        points = points[vis_ids] # NOTE: vis_ids is boolean in pts_set_ids
        normals = normals[vis_ids] # NOTE: vis_ids is boolean in pts_set_ids
        mm_per_pixel = calculate_mm_per_pixel_per_camera(camera_data, points, normals, camera_pose, unit_length=10, S=50)

        pts_set_resolution.append(mm_per_pixel)

    resolution_per_point = np.zeros((len(pts)))
    for pts_set_ids, mm_per_pixel in zip(pts_set_per_camera, pts_set_resolution):
        if len(pts_set_ids) == 0:
            continue
        resolution_per_point[pts_set_ids] = mm_per_pixel

    return resolution_per_point