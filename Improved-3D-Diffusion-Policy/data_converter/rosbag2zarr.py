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

def process_point_cloud(camera_info_msgs, rgb_images, depth_images):
    all_point_clouds = []
    for index in range(len(camera_info_msgs)):
        camera_info = camera_info_msgs[index]
        rgb_image = rgb_images[index]
        depth_image = depth_images[index]
        camera_intrinsics = np.array(camera_info.K).reshape(3, 3)
        point_cloud = create_point_cloud(rgb_image, depth_image, camera_intrinsics)
        all_point_clouds.append(point_cloud)
    # print("all_point_clouds_shape", all_point_clouds.shape)
    return all_point_clouds

def get_extrinsics_matrix(camera_info):
    R = np.reshape(camera_info.R, (3, 3))
    P = np.reshape(camera_info.P, (3, 4))
    extrinsics_matrix = np.hstack((R, P[:, 3].reshape(3, 1)))
    extrinsics_matrix = np.vstack((extrinsics_matrix, np.array([0, 0, 0, 1])))
    return extrinsics_matrix

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

save_data_path = '/home/a/Projects/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/real_data/test.zarr'
demo_dirs = read_file_to_list("/home/a/Projects/3D-Diffusion-Policy/3D-Diffusion-Policy/data_loader/skin_bag_file_list_small.txt")
print("\n".join(demo_dirs))

# storage
total_count = 0
img_arrays = []
point_cloud_arrays = []
depth_arrays = []
state_arrays = []
action_arrays = []
episode_ends_arrays = []


if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = 'y'
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)

for demo_dir in demo_dirs:
    dir_name = os.path.dirname(demo_dir)

    cprint('Processing {}'.format(demo_dir), 'green')
    reader = RosbagReader(demo_dir)
    reader.align_all_msgs('/camera/color/image_raw')

    rgb_images = reader.export_aligned_images('/camera/color/image_raw')
    depth_images = reader.export_aligned_images('/camera/depth/image_raw')
    robot_states = np.concatenate([ \
                       np.array(reader.export_aligned_positions('/human_arm_ctrl_left')), \
                       np.array(reader.export_aligned_positions('/human_arm_ctrl_right')),\
                       np.array(reader.export_aligned_positions('/inspire_hand/ctrl/left_hand')), \
                       np.array(reader.export_aligned_positions('/inspire_hand/ctrl/right_hand'))], axis=1)
    
    print("robot_states_shape", robot_states.shape)
    actions = [robot_states[i+1] - robot_states[i] for i in range(len(robot_states)-1)] + [np.zeros_like(robot_states[0])]
    camera_info_msgs = reader.export_aligned_msgs('/camera/color/camera_info')
    point_clouds = process_point_cloud(camera_info_msgs, rgb_images, depth_images)
    
    demo_length = len(point_clouds)
    for step_idx in tqdm.tqdm(range(demo_length)):
       
        total_count += 1
        obs_image = rgb_images[step_idx]
        obs_depth = depth_images[step_idx]
        obs_image = preproces_image(obs_image)
        obs_depth = preproces_image(np.expand_dims(obs_depth, axis=-1)).squeeze(-1)
        obs_camera_info = camera_info_msgs[step_idx]
        obs_extrinsics_matrix = get_extrinsics_matrix(obs_camera_info)
        obs_pointcloud = point_clouds[step_idx]
        robot_state = robot_states[step_idx]
        action = actions[step_idx]
    
        obs_pointcloud = preprocess_point_cloud(obs_pointcloud, obs_extrinsics_matrix, use_cuda=True)
        print("debug obs_pointcloud shape", obs_pointcloud.shape)
        img_arrays.append(obs_image)
        action_arrays.append(action)
        point_cloud_arrays.append(obs_pointcloud)
        depth_arrays.append(obs_depth)
        state_arrays.append(robot_state)
    
    episode_ends_arrays.append(total_count)

# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

img_arrays = np.stack(img_arrays, axis=0)
if img_arrays.shape[1] == 3: # make channel last
    img_arrays = np.transpose(img_arrays, (0,2,3,1))
point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
depth_arrays = np.stack(depth_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    raise NotImplementedError
zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# print shape
cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

