import os
import cv2
import pickle
import numpy as np
import trimesh
import open3d as o3d
import Camera

def init_data(mesh_path, sampled_pts_path, camera_pkl_file, ):

    # load mesh
    mesh_trimesh = trimesh.load(mesh_path, process=False, use_embree=True)

    # load points: uniformly sampled on the mesh
    pcd = o3d.io.read_point_cloud(sampled_pts_path)
    pts = np.asarray(pcd.points)
    pts_normals = np.asarray(pcd.normals)

    # load camera configuration
    cameras_network = Camera.loadCamerasNetwork(camera_pkl_file)
    cameras = cameras_network.cameras
    camera_positions, camera_dirs, camera_poses = cameras_network.parseCameras()

    # create the visible point set for each camera based on FOV and considering occlusion
    vis_pts_set_per_camera, pts_set_ids_visible = getVisiblePointSetAllCameras(mesh_trimesh, pts, cameras, camera_positions, camera_poses, return_unique_set=True) # list of vertex indices in trimesh
    print("num of visible pts: ", len(pts_set_ids_visible))

    # update points and vis_pts_set_per_camera after removing invisible points
    pts = pts[pts_set_ids_visible]
    pts_normals = pts_normals[pts_set_ids_visible]
    vis_pts_set_per_camera, temp = getVisiblePointSetAllCameras(mesh_trimesh, pts, cameras, camera_positions, camera_poses, return_unique_set=True) # list of vertex indices in trimesh
    assert len(pts_set_ids_visible) == len(temp)

    return  pts, pts_normals, vis_pts_set_per_camera, cameras_network

def save_output(cameras_network, focus_distances, max_pts_set_per_camera, pts, pts_normals,\
                pts_set_per_camera=None, outfname="output.pkl", save_dir=""):
    output_data = {}
    output_data["cameras_network"] = cameras_network 
    output_data["focus_distances"] = focus_distances
    output_data["max_pts_set_per_camera"] = max_pts_set_per_camera
    output_data["pts"] = pts # sampled
    output_data["pts_normals"] = pts_normals
    output_data["pts_set_per_camera"] = pts_set_per_camera # assigned

    outfname = os.path.join(save_dir, outfname)
    with open(outfname, "wb") as outfile:
        pickle.dump(output_data, outfile, pickle.HIGHEST_PROTOCOL)
    print(outfname + " is saved.")

def load_output(outfname):
    with open(outfname, "rb") as infile:
        output_data = pickle.load(infile)

    cameras_network = output_data["cameras_network"]
    focus_distances = output_data["focus_distances"]
    max_pts_set_per_camera = output_data["max_pts_set_per_camera"]
    pts = output_data["pts"]
    pts_normals = output_data["pts_normals"]
    pts_set_per_camera = output_data["pts_set_per_camera"]

    return (cameras_network, focus_distances, max_pts_set_per_camera, pts, pts_normals,\
            pts_set_per_camera)

def wrap_kview_result(losses_camera_group_indiv, losses_camera_group_common, focus_distances_camera_group, assignment_camera_group):

    result = {}
    result["losses_camera_group_indiv"] = np.array(losses_camera_group_indiv)
    result["losses_camera_group_common"] = np.array(losses_camera_group_common )
    result["focus_distances_camera_group"] = np.array(focus_distances_camera_group)
    result["assignment_camera_group"] = np.array(assignment_camera_group, dtype='object')

    return result

def get_kview_result(result):
    losses_camera_group_indiv = result["losses_camera_group_indiv"]
    losses_camera_group_common = result["losses_camera_group_common"]
    focus_distances_camera_group = result["focus_distances_camera_group"]
    assignment_camera_group = result["assignment_camera_group"]

    return losses_camera_group_indiv, losses_camera_group_common, focus_distances_camera_group, assignment_camera_group

##################################################################
# Utils for Mesh (target obejct) and Cameras related functions
##################################################################

def getAreaOfSubset(mesh, vertex_indices):
    """
        Get surface area of subset (input vertices) of a mesh
    Args:
        mesh: custom mesh class
        vertex_indices:
    Returns:
        accumulated_area:
    """

    # Make a set of vertex indices
    vertex_indices = list(vertex_indices)

    # # Slow version
    # accumulated_area = 0
    # for i, face in enumerate(mesh.faces):
    #     if not vertex_indices.issuperset(set(face)): # vertex_indices should be type set
    #         continue
    #     accumulated_area += face_areas[i]

    # Vectorization version
    face_ids = np.all(np.isin(mesh.faces, vertex_indices), axis=1)
    accumulated_area = np.sum(mesh.area_faces[face_ids])
    return accumulated_area

def getUniqueElements(arr):
    unique_arr = []
    for ele in arr:
        unique_arr.extend(list(ele))
    unique_arr = np.unique(np.array(unique_arr))
    unique_arr = np.sort(unique_arr)
    return unique_arr

def initialize_focus_distances(pts, pts_set_per_camera, camera_positions, camera_dirs, mode='mean'):
    """
        Initialize focus distance for each camera.

        Use zero/closest/average depth from the camera to its viisible point set.

    Args:
        pts: total point set, np array of shape (num points x 3) (in cartesian coordinates)
        pts_set_per_camera: list of point set for each camera. Each point set is represented in point ids, 1-d np array.
                            NOTE: Ids are ordered in the order of the total point set.
                            NOTE: The points are already processed so that all the points are visible to the camera
        camera_positions: array of camera positions, np array of shape (num cameras x 3)
        camera_dirs: array of camera viewing directions, np array of shape (num cameras x 3)
        mode: how to initialize focus distance, "min" for closest depth, "mean" for average depth, "zero" for zero depth

    Returns:
        focus_distances: array of focus distances, np array of shape (num cameras, )
    """

    focus_distances = []
    for i, pts_set_ids, camera_position, camera_dir in zip(range(len(camera_positions)), pts_set_per_camera, camera_positions, camera_dirs):
        if len(pts_set_ids) == 0: # no points visibile to this camera  # NOTE: check if there is a better way to handle this
            focus_distances.append(0)
            print("[Warning]: No point is visible to camera {}".format(i))
            continue

        pts_set_ids = np.array(pts_set_ids)
        points = pts[pts_set_ids]
        depths = trimesh.points.point_plane_distance(points, camera_dir.flatten(), camera_position.flatten())

        if mode == "min": # closest depth
            focus_distance = np.min(depths)
        elif mode == "mean": # average depth
            focus_distance = np.mean(depths)
        elif mode == "zero": # zero depth
            focus_distance = 0
        else:
            raise ValueError("Invalid mode: ", mode)
        focus_distances.append(focus_distance)

    focus_distances = np.array(focus_distances)

    return focus_distances

def getVisiblePointSetSingleCamera(mesh, pts, camera_position, camera_pose, w, h, K, distortion, intersection_tolerance=0.001):
    """
    Get visible point set for a single camera based on its FOV and occlusion.

    Args:
        mesh: trimesh object
        pts: point set 1, np array of shape (num points x 3)
        camera_position: list of camera positions, np array of shape (num cameras x 3)
        camera_pose: list of camera poses, list of np array of shape (4 x 4)
        w: image width
        h: image width
        K: camera intrinsic matrix
        distortion: camera distortion coefficients
        intersection_tolerance: tolerance for intersection to check occlusion
    Returns:
        valid_pt_ids: list of valid point ids (1-d np array)
    """

    H_cameraToModel = np.linalg.inv(camera_pose)
    # NOTE: fix shape of location from (3,1) to (1,1,3) for opencv4.1.0
    # NOTE: img_coord in (pixel_x, pixel_y), corresponding to (w,h) in the image
    img_coords ,__ =cv2.projectPoints(pts.reshape((-1,1,3)), H_cameraToModel[:3,:3], H_cameraToModel[:3,3], K, distortion)
    # test by FOV
    img_coords = img_coords.reshape((-1,2))
    valid_pt_ids = (img_coords[:,0] > 0) & (img_coords[:,0] < w ) \
                & (img_coords[:,1] > 0) & (img_coords[:,1] < h)
    valid_pt_ids = np.nonzero(valid_pt_ids)[0]

    # By occlusion (checking by ray intersection with mesh)
    ray_origins = np.tile(camera_position.reshape((-1,3)), (len(valid_pt_ids), 1))
    ray_directions = pts[valid_pt_ids] - ray_origins
    # indices_tri, _ = mesh.ray.intersects_id(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
    locations, ray_indices , _ = mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
    diff = np.linalg.norm(locations - pts[valid_pt_ids][ray_indices], axis=1) # note: use ray_indices for the correct order of points for comparison
    valid_pt_ids = valid_pt_ids[ray_indices][diff<intersection_tolerance] # note: reordering of valid_pt_ids is done by ray_indices

    return valid_pt_ids

def getVisiblePointSetAllCameras(mesh, pts, cameras, camera_positions, camera_poses, return_unique_set=False):
    """
        Get visible point set for all cameras, based on individual FOV and occlusion.
    Args:
        mesh: trimesh object
        pts: point set 1, np array of shape (num points x 3)
        camera_position: list of camera positions, np array of shape (num cameras x 3)
        camera_pose: list of camera poses, list of np array of shape (4 x 4)
    Returns:
        pts_set_per_camera: list of set of points for each camera (in point ids, 1-d np array)
    """

    pts_set_per_camera = []  # list of set of points for each camera (in point ids)
    w = cameras[0].img_resolution[0]
    h = cameras[0].img_resolution[1]
    K = cameras[0].intrinsics.K
    distortion = cameras[0].intrinsics.dist

    for camera_position, camera_pose in zip(camera_positions, camera_poses):
        pts_set_ids = getVisiblePointSetSingleCamera(mesh, pts, camera_position, camera_pose, w, h, K, distortion)
        pts_set_per_camera.append(pts_set_ids)

    if return_unique_set:
        pts_set_ids_all = getUniqueElements(pts_set_per_camera)
        return pts_set_per_camera, pts_set_ids_all

    return pts_set_per_camera

def getVisibilityListPerPoint(pts, pts_set_per_camera):
    """
        For each point, associate a list of cameras that can see it 
        camera_id: 0, 1, 2, ...
    """
    visibility_list_per_point = []
    for pt_id, pt in enumerate(pts):
        visibility_list = []
        for camera_id, pts_set in enumerate(pts_set_per_camera):
            if pt_id in pts_set:
                visibility_list.append(camera_id)
        visibility_list_per_point.append(visibility_list)

    return visibility_list_per_point

def getPlaneIntersectionPointsAndNormals(mesh, plane_origin, plane_normal, frustum_planes):
    """
        Get the view quality scores for each point in pts for each camera. The score is a weighted sum of the distance and alignment scores.
        Distance score: depth to the camera, or the depth to the focus plane of the camera, normlalized by the largest depth value
        Alignment score: normalized dot product between the normal vector of the point and the camera direction

    Args:
        mesh: point set 1, np array of shape (num points x 3)
        plane_origin: normal vector associated with each point in pts, np array of shape (num points x 3)
        plane_normal: list of camera positions, np array of shape (num cameras x 3)
        frustum_planes: list of camera directions, np array of shape (num cameras x 3)
    Returns:
        pts_inside: np array of shape (num points x 3)
        interp_normals: np array of shape (num points x 3)
    """
    # get intersection
    # lines, sections_3d, _ = trimesh.intersections.mesh_multiplane(mesh_trimesh, plane_origin=center, plane_normal=z, heights=[dof_n, dof_f])
    slice = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if slice is None:
        return (None, None)

    # # get points of contours within the frustum (slow and incorrect) # TODO: get closed contours at the planes/edges of the frustum, do we need the contours to be closed?
    # frustum_triangulated = frustum.triangulate()
    # frustum_trimesh = trimesh.Trimesh(vertices=frustum_triangulated.points,\
    #                         faces=frustum_triangulated.faces.reshape(frustum_triangulated.n_faces, 4)[:,1:], use_embree=True) # debug
    # pts_inside_ids = frustum_trimesh.contains(slice.vertices) # Get vertices of the mesh inside the frustum
    # pts_inside = slice.vertices[pts_inside_ids]

    # Get points of contours inside the frustum
    left_plane = np.asarray(frustum_planes[0:4]).reshape((4, 1))
    right_plane = np.asarray(frustum_planes[4:8]).reshape((4, 1))
    bottom_plane = np.asarray(frustum_planes[8:12]).reshape((4, 1))
    top_plane = np.asarray(frustum_planes[12:16]).reshape((4, 1))
    pts = np.hstack((slice.vertices, np.ones((len(slice.vertices), 1))))
    pts_inside_ids = (np.matmul(pts, left_plane) >= 0).flatten() & (np.matmul(pts, right_plane) >= 0).flatten()  &\
                    (np.matmul(pts, bottom_plane) >= 0).flatten() & (np.matmul(pts, top_plane) >= 0).flatten()
    pts_inside = slice.vertices[pts_inside_ids]
    if len(pts_inside) == 0:
        return (None, None)

    # get normal of the points
    (closest_points, distances, triangle_ids) = mesh.nearest.on_surface(pts_inside) 
    if np.max(distances) > 0.000001: # if not np.all(closest_points == pts_inside): # TODO: why are some points not on the surface?
        print("closest points not equal query points: ", np.max(distances))
    bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[triangle_ids], points=pts_inside)
    interp_normals = trimesh.unitize((mesh.vertex_normals[mesh.faces[triangle_ids]] *
                              trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
    
    return (pts_inside, interp_normals)

def getVisiblePtsOverlapsAvg(pts, camera_ids_group, vis_pts_set_per_camera, return_overlaps_ratio_per_camera_pair=False):
    '''
        Get average overlaps of visible points between camera pairs:
            For each camera pair, the overlaps is: number of overlapping points / number of visible points
    '''

    overlaps_avg = 0
    overlaps_ratio_per_camera_pair = []
    for camera_ids in camera_ids_group:
        camera_0, camera_1 = camera_ids
        vis_pts_0 = vis_pts_set_per_camera[camera_0]
        vis_pts_1 = vis_pts_set_per_camera[camera_1]
        intersection = np.intersect1d(vis_pts_0, vis_pts_1)
        overlaps = len(intersection)
        overlaps_ratio_all = overlaps / len(pts)
        overlaps_avg += overlaps_ratio_all
        overlaps_ratio_per_camera_pair.append(overlaps_ratio_all)

        # overlaps_ratio_0 = overlaps / len(vis_pts_0)
        # overlaps_ratio_1 = overlaps / len(vis_pts_1)
        # print("Overlaps all: {} ({}), overlaps camera {}: {} ({}), overlaps camera {}: {} ({})".format(overlaps_ratio_all, overlaps,\
        #                         camera_0, overlaps_ratio_0, len(vis_pts_0), camera_1, overlaps_ratio_1, len(vis_pts_1)) )
        # overlaps_avg += (overlaps_ratio_0 + overlaps_ratio_1) / 2
    overlaps_avg /= len(camera_ids_group)
    overlaps_ratio_per_camera_pair = np.array(overlaps_ratio_per_camera_pair)
    # print("Overlaps average: {}".format(overlaps_avg))
    if return_overlaps_ratio_per_camera_pair:
        return overlaps_avg, overlaps_ratio_per_camera_pair
    return overlaps_avg