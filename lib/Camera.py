#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: weilunhuang
"""
import json
import numpy as np
import pickle
import trimesh
import pyvista as pv
import networkx as nx
import itertools

class Intrinsics(object):
    ''' A class for intrinsic parameters'''
    def __init__(self, fx, fy, px, py, dist):
        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
        self.K = np.array([[fx, 0, px],[0, fy, py],[0, 0, 1]])
        self.dist = dist

class CameraData(object):
    ''' A class for filename, img resolution, intrinsic parameters, and extrinsic parameters of camera'''
    def __init__(self, fname, img_resolution, intrinsics, center = np.zeros((3,1)), orientation = np.eye(3)):
        self.fname =fname
        self.img_resolution = img_resolution
        self.intrinsics = intrinsics 
        self.center = center
        self.orientation = orientation
        pose = np.concatenate((orientation, center.reshape((3,1))),axis=1)
        pose = np.concatenate((pose,np.array([0,0,0,1]).reshape((1,4))),axis=0)
        self.pose = pose

class Camera(object):
    ''' A class for intrinsic parameters and extrinsic parameters of camera, modified to accomodate different kinds of cameras in a scan'''
    def __init__(self, path):
        # read json info for camera
        self.json_path = path
        with open(path) as json_file:
            self.cam_info=json.load(json_file)
        self.cameras = []

        # construct intrinsic and extrinsic parameters
        self.constructParameters()
    
    def constructParameters(self):

        for i, view in enumerate(self.cam_info['views']):

            filename = view['value']['ptr_wrapper']['data']['filename']
            pose_id = view['value']['ptr_wrapper']['data']['id_pose']
            intrinsic_id = view['value']['ptr_wrapper']['data']['id_intrinsic']

            img_resolution = (view['value']['ptr_wrapper']['data']['width'],\
                            view['value']['ptr_wrapper']['data']['height'])

            # construct intrinsics
            focal_length = self.cam_info['intrinsics'][intrinsic_id]['value']['ptr_wrapper']['data']['focal_length']
            px = self.cam_info['intrinsics'][intrinsic_id]['value']['ptr_wrapper']['data']['principal_point'][0]
            py = self.cam_info['intrinsics'][intrinsic_id]['value']['ptr_wrapper']['data']['principal_point'][1]
            # dist = self.cam_info['intrinsics'][intrinsic_id]['value']['ptr_wrapper']['data']['disto_k3']
            # dist.extend([0.0, 0.0])
            dist = [0.0, 0.0, 0.0, 0.0, 0.0]
            dist = np.array(dist)
            intrinsics = Intrinsics(focal_length, focal_length, px, py, dist)

            # # construct extrinsics
            # center = np.array(self.cam_info['extrinsics'][pose_id]['value']['center']).reshape((3,1)) # make the center as a column vector
            # orientation = np.array(self.cam_info['extrinsics'][pose_id]['value']['rotation']).T # rotation matrix from openMVG is column-major
            # self.cameras.append(CameraData(filename, img_resolution, intrinsics, center, orientation))

            # construct extrinsics
            # NOTE: Add for missing camera poses
            if self.cam_info['extrinsics'][pose_id]['value']['center'] is not None:
                center = np.array(self.cam_info['extrinsics'][pose_id]['value']['center']).reshape((3,1)) # make the center as a column vector
                orientation = np.array(self.cam_info['extrinsics'][pose_id]['value']['rotation']).T # rotation matrix from openMVG is column-major
                self.cameras.append(CameraData(filename, img_resolution, intrinsics, center, orientation))
            else:
                self.cameras.append(CameraData(filename, img_resolution, intrinsics))
                print("[WARNING] Missing camera pose for frame: " + filename)
                print("Using (0,0,0) and Identity matrix for its center and orientation.")

    def getCameraIDFromFname(self, fname):
        """ Function to get camera id from a frame name
            Args:
                fname: frame name, string.
            Returns:
                camera id: corresponding camera for the frame name, int.
        """
        for i, view in enumerate(self.cam_info['views']):
            if fname in view['value']['ptr_wrapper']['data']['filename']:
                return i

    def getCameraFromFname(self, fname):
        """ Function to get camera data from a frame name
            Args:
                fname: frame name, string.
            Returns:
                camera: corresponding camera for the frame name, CameraData.

        """
        for i, view in enumerate(self.cam_info['views']):
            if fname == view['value']['ptr_wrapper']['data']['filename']:
                return self.cameras[i]

        # case for incorrect fname
        return None 

    def getFnameFromCameraID(self, camera_id):
        """ Function to get frame name from camera id
            Args:
                camera id: camera id, int.
            Returns:
                fname: corresponding frame name for the camera id, string.
        """
        return self.cameras[camera_id].fname

class CamerasNetwork(object):
    def __init__(self, cameras_config=None, cameras=None, cameras_topology=None):
        self.cameras_config = cameras_config # (num_cirlces, num_pt_per_circle, normal_offset, radius)
        self.cameras = cameras # list of CameraData
        self.cameras_topology = cameras_topology # networkx graph

    def parseCameras(self):
        """
        Function to parse cameras for camera positions, cmaera directions, and camera poses
        Return:
        
        """
        camera_positions = []
        camera_poses = []
        camera_dirs = []

        for camera in self.cameras:
            pose = camera.pose
            center = pose[:3,3]
            normal = pose[:3,2]
            camera_positions.append(center)
            camera_dirs.append(normal)
            camera_poses.append(pose)

        camera_positions = np.array(camera_positions).reshape((-1,3))
        camera_dirs = np.array(camera_dirs).reshape((-1,3))
        camera_poses = np.array(camera_poses)

        return (camera_positions, camera_dirs, camera_poses)

    def createCameraPairsID(self, ring=(1,)):
        """
        Function to create camera pairs ID
        Return:
            camera_pairs_id: list of camera pairs ID
        """
        camera_pairs_id = []

        if 1 in ring: # 1-ring
            for edge in self.cameras_topology.edges():
                camera_pairs_id.append((edge[0], edge[1]))
        if 2 in ring: # 2-ring
            edges_visited = set() # avoid duplicate edges
            for node in self.cameras_topology.nodes():
                for node_d in nx.descendants_at_distance(self.cameras_topology, node, 2):
                    if (node, node_d) in edges_visited:
                        continue
                    camera_pairs_id.append((node, node_d))
                    edges_visited.add((node, node_d))
                    edges_visited.add((node_d, node))

        return camera_pairs_id

    def createCameraParrsMaximalIndependentSets(self, ring=(1,)):
        """
        Function to create camera pairs ID
        Return:
            camera_pairs_id: list of camera pairs ID
        """
        (num_cirlces, num_pt_per_circle, _, _) = self.cameras_config
        # label horizontal and vertical edges
        for edge in self.cameras_topology.edges():
            if abs(edge[0] - edge[1]) < num_pt_per_circle:
                self.cameras_topology[edge[0]][edge[1]]['type'] = 'horizontal'
            else:
                self.cameras_topology[edge[0]][edge[1]]['type'] = 'vertical'
        # print(self.cameras_topology.edges(data=True))

        sets = []
        if 1 in ring: # 1-ring
             
            set_1 = [] # horizontal: (2i,j) to (2i+1,j)
            set_2 = [] # horizontal: (2i+1,j) to (2i+2,j)
            set_3 = [] # vertical: (i,2j) to (i,2j+1)
            set_4 = [] # vertical: (i,2j+1) to (i,2j+2)
            for edge in self.cameras_topology.edges():
                # horizontal
                if self.cameras_topology[edge[0]][edge[1]]['type'] == 'horizontal':
                    if edge[0] % 2 == 0 and edge[1] == edge[0] + 1:
                        set_1.append((edge[0], edge[1]))
                    else:
                        set_2.append((edge[0], edge[1]))
                # vertical
                if self.cameras_topology[edge[0]][edge[1]]['type'] == 'vertical':
                    if edge[0]//num_pt_per_circle % 2 == 0: # using the fact the smaller index is alwasys the first one in the pair
                        set_3.append((edge[0], edge[1]))
                    else:
                        set_4.append((edge[0], edge[1]))

            sets.append(set_1)
            sets.append(set_2)
            sets.append(set_3)
            sets.append(set_4)

            # print("SET 1: ", len(set_1))
            # print(set_1)
            # print("SET 2: ", len(set_2))
            # print(set_2)
            # print("SET 3: ", len(set_3))
            # print(set_3)
            # print("SET 4: ", len(set_4))
            # print(set_4)
            # print("len of edges preserves:")
            # print(len(set_1) + len(set_2) + len(set_3) + len(set_4) == len(self.cameras_topology.edges()))

        if 2 in ring: # 2-ring
            raise NotImplementedError

        return sets

    def createCameraTripletID(self, ring=(1,)):
        """
        Function to create camera triplet ID
        # NOTE: Only support 1 ring
        # TODO: check if there are duplicate triplets (e.g. (1,2,3) and (3,2,1))
        Return:
            camera_triplets_id: list of camera pairs ID
        """
        camera_triplets_id = set()

        if 1 in ring: # 1-ring
            for node in self.cameras_topology.nodes():
                nodes = nx.descendants_at_distance(self.cameras_topology, node, 1)
                nodes.update([node])
                triplets = list(itertools.combinations(nodes, 3))
                camera_triplets_id.update(triplets)
        camera_triplets_id = list(camera_triplets_id)

        return camera_triplets_id

def loadCamerasNetwork(camera_pickle_file):
    cameras_network = pickle.load(open(camera_pickle_file, "rb"))
    return cameras_network
    
#####################
#### DOF related ####
#####################

def dof_cal(focus_distance, focal_length=50, hyperfocal_distance=10000):
    """
        Calculate near and far depth of field limit given the focus distance, focal length and hyperfocal distance.
        NOTE: This is a simplified version of the calculation with focal length and hyperfocal distance
    Args:
        focal_distance: 
        focal_length: 50
        H (hyperfocal distance): 10000, 4000, 2000
    Returns:
        dof_n: distance from camera to near depth of field limit
        dof_f: distance from camera to far depth of field limit
    """
    H = hyperfocal_distance

    dof_n = H * focus_distance / (H + focus_distance - focal_length)
    dof_f = H * focus_distance / (H - focus_distance + focal_length)

    return (dof_n, dof_f)


def focus_interval_cal(depth, focal_length=50, hyperfocal_distance=10000):
    """
        Calculate near and far limit distance so that the point at givien depth can be covered in focus.
        NOTE: This is a simplified version of the calculation with focal length and hyperfocal distance.
        NOTE: Use dof_f equantion for the near limit and dof_n equation for the far limit.
    Args:
        depth: 
        focal_length: 50
        hyperfocal_distance: 10000
    Returns:
        s_0: near limit distance using dof_f equation
        s_1: far limit distance using dof_n equation
    """
    # focal_length = FOCAL_LENGTH
    # H = HYPER_FOCAL_DISTANCE

    s_0 = depth * (hyperfocal_distance  + focal_length) / (hyperfocal_distance + depth)
    s_1 = depth * (hyperfocal_distance  - focal_length) / (hyperfocal_distance - depth)
    
    # NOTE: s_0 is the near limit and s_1 is the far limit.

    # # TODO: should be adjusted based on the solution between s_0 and s_1, should align with getFocalLoss in Loss.py
    # s_0 += 0.5 * (depth - s_0)
    # s_1 -= 0.5 * (s_1 - depth)

    return (s_0, s_1)

def get_pts_in_dof(pts, camera_position, camera_dir, focus_distance, focal_length=50, hyperfocal_distance=10000):
    """
        Filter points based on depth of field.
    Returns:
        pts_in_dof:
        num_pts:
    """

    depths = trimesh.points.point_plane_distance(pts, camera_dir.flatten(), camera_position.flatten())
    dof_n, dof_f = dof_cal(focus_distance, focal_length, hyperfocal_distance)
    pts_in_dof = (depths > dof_n) & (depths < dof_f)
    num_pts = np.sum(pts_in_dof)
    pts_in_dof = np.nonzero(pts_in_dof)[0] # index of points

    return (pts_in_dof, num_pts)

def get_pts_in_dof_mask(pts, camera_position, camera_dir, focus_distance, focal_length=50, hyperfocal_distance=10000):
    """
        Filter points based on depth of field.
    Returns:
        mask:
    """

    depths = trimesh.points.point_plane_distance(pts, camera_dir.flatten(), camera_position.flatten())
    dof_n, dof_f = dof_cal(focus_distance, focal_length, hyperfocal_distance)
    mask = (depths > dof_n) & (depths < dof_f)

    return mask

def get_pts_out_of_dof(pts, camera_position, camera_dir, focus_distance, focal_length=50, hyperfocal_distance=10000):
    """
        Filter points based on depth of field.
    Returns:
        pts_out_of_dof:
        num_pts:
    """

    depths = trimesh.points.point_plane_distance(pts, camera_dir.flatten(), camera_position.flatten())
    dof_n, dof_f = dof_cal(focus_distance, focal_length, hyperfocal_distance)
    pts_in_dof = (depths > dof_n) & (depths < dof_f)
    pts_out_of_dof = ~pts_in_dof
    num_pts = np.sum(pts_out_of_dof)
    pts_out_of_dof = np.nonzero(pts_out_of_dof)[0]

    return (pts_out_of_dof, num_pts)

def createFrustum(camera_pose, view_angle, focus_distance, aspect_ratio=0.666666666, focal_length=50, hyperfocal_distance=10000):
    """
        Get surface area of subset (input vertices) of a mesh
    Args:
        mesh: custom mesh class
        vertex_indices:
    Returns:
        frustum:
        frustum_planes:
    """
    # Construct camera frustum based on FOV and DOF

    camera_position = camera_pose[:3,3]
    y = camera_pose[:3,1]
    z = camera_pose[:3,2]

    camera_pv = pv.Camera()
    (dofn, doff) = dof_cal(focus_distance, focal_length, hyperfocal_distance)
    near_range = dofn 
    far_range = doff
    focal_point = camera_position + z * focus_distance
    camera_pv.clipping_range = (near_range, far_range)
    camera_pv.position = camera_position 
    camera_pv.up = -y
    camera_pv.focal_point = focal_point
    # Set vertical view angle as an indirect way of setting the y focal distance
    camera_pv.SetViewAngle(view_angle)
    frustum = camera_pv.view_frustum(aspect=aspect_ratio)
    frustum_planes = [0] * 24 # left,right,bottom,top,far,near: normals point inward
    camera_pv.GetFrustumPlanes(aspect_ratio, frustum_planes)

    return (frustum, frustum_planes)

def test():
    # print("test: dof_cal")
    # print(dof_cal(0))
    # print(focus_interval_cal())
    # print(focus_interval_cal(500))
    # print(dof_cal(478.57142857142856))
    # print(dof_cal(523.6842105263158))

    # np.random.seed(0)
    # depths = np.random.randint(100, 500, size=10)
    # s_0, s_1 = focus_interval_cal(depths)
    # t_0 = np.column_stack((s_0, np.ones(s_0.shape)))
    # t_1 = np.column_stack((s_1, -np.ones(s_0.shape)))
    # t = np.vstack((t_0, t_1))
    # t = t[np.argsort(t[:,0], axis=0)]

    # # chi = np.cumsum(t[:,1]).astype(int)
    # # id_opt = np.argmax(chi)
    # # f = (t[id_opt, 0] + t[id_opt + 1, 0]) / 2

    # print("intervals:")
    # print(t[:,0])
    # midpoints = (t[:-1,0] + t[1:,0]) / 2
    # print("midpoints")
    # print(midpoints)

    camera_pkl_file = "../data/3dbodytex/000/camera_poses_cylinder_5_8_400_500.pkl"
    # cameras
    cameras_network = loadCamerasNetwork(camera_pkl_file)
    cameras = cameras_network.cameras
    camera_positions, camera_dirs, camera_poses = cameras_network.parseCameras()

    camera_positions, camera_dirs, _ = cameras_network.parseCameras()
    RING = (1,)
    camera_ids_group = cameras_network.createCameraPairsID(ring=RING)
    print(camera_ids_group)

    # sets = nx.maximal_independent_set(cameras_network.cameras_topology)
    # print(sets)
    print("===============")
    cameras_network.createCameraParrsMaximalIndependentSets(ring=RING)

if __name__ == "__main__":
    test()