import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time


import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle
from rosbag_reader import RosbagReader

# import visualizer

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(point_cloud, extrinsics_matrix, use_cuda=True):
    points = np.asarray(point_cloud.points)

    num_points = 1024    

    # scale
    point_xyz = points[..., :3]
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix)
    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz

    # crop if needed
    # points = points * 0.0002500000118743628
    # WORK_SPACE = [
    #     [0.65, 1.1],
    #     [0.45, 0.66],
    #     [-0.7, 0]
    # ]
    # points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
    #                             (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
    #                             (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

    # visualizer.visualize_pointcloud(points)

    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    # print(points.shape)
    return points
   
def preproces_image(image):
    img_size = 84
    
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def create_point_cloud(rgb_img, depth_img, camera_intrinsics):
    rgb_o3d = o3d.geometry.Image(rgb_img)
    depth_o3d = o3d.geometry.Image(depth_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=rgb_img.shape[1],
        height=rgb_img.shape[0],
        fx=camera_intrinsics[0, 0],
        fy=camera_intrinsics[1, 1],
        cx=camera_intrinsics[0, 2],
        cy=camera_intrinsics[1, 2]
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

def get_extrinsics_matrix(camera_info):
    R = np.reshape(camera_info.R, (3, 3))
    P = np.reshape(camera_info.P, (3, 4))
    extrinsics_matrix = np.hstack((R, P[:, 3].reshape(3, 1)))
    extrinsics_matrix = np.vstack((extrinsics_matrix, np.array([0, 0, 0, 1])))
    return extrinsics_matrix

if __name__ == '__main__':
    # more details see Improved-3D-Diffusion-Policy/data_converter/rosbag_reader.py
    # rgb_images = reader.export_aligned_images('/camera/color/image_raw')
    # depth_images = reader.export_aligned_images('/camera/depth/image_raw')
    
    point_cloud = create_point_cloud(rgb_img, depth_img, camera_intrinsics)
    obs_extrinsics_matrix = get_extrinsics_matrix(obs_camera_info)
    obs_pointcloud = preprocess_point_cloud(point_cloud, obs_extrinsics_matrix, use_cuda=True)
    # obs_point_cloud is all u need.