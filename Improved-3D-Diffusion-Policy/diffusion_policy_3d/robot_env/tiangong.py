import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.common.gr1_action_util as action_util
import diffusion_policy_3d.common.rotation_util as rotation_util
import diffusion_policy_3d.common.point_cloud_util as point_cloud_util

import tqdm
import torch
import os 
import cv2
import numpy as np
import sys
import torchvision

# Add the project directory to the Python path
project_path = str(pathlib.Path(__file__).parent.parent)
sys.path.append(project_path)

# pro 4
sys.path.append('/home/ps/Devs/vibrant/inrocs/inrocs')

initial_state = [-0.14561805, 0.06069489, -0.51761932, -0.18, 0.28581739, -0.10982346, -0.13352916, 0.26115257, -0.10885823, 1.35870292, 1.07909285, -0.98161446, -0.14459579, 0.24475515, 0.82264018, 0.79090095, 0.78081647, 0.65218203, 0.47389977, 0.77681202, 0.92088979, 0.83451309, 0.8894613, 0.88955654, 0.588442, 0.90074142]

class TiangongDexEnvInference:
    """
    The deployment is running on the local computer of the robot.
    """
    def __init__(self, obs_horizon=2, action_horizon=8, device="gpu",
                use_point_cloud=True, use_image=True, img_size=84,
                num_points=1024,camera_names=['left'],debug_mode=True):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.img_size=img_size
        self.num_points=num_points
        self.camera_names=camera_names
        self.debug_mode=debug_mode
        
        self.action_array = []
        self.cloud_array = []
        self.color_array = []
        self.depth_array = []
        self.env_qpos_array = []
        
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
    
    def step(self, action_list):
        
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            
            # de normalize action
            action_pred = self.process_action(act)
            
            if self.debug_mode:
                obs={"qpos": np.zeros(26), "images": {"left": np.zeros((480, 640, 3))}}
            else:
                from robot_env.tianyi_env import tianyi_env as robot_env
                robot_env.step_full(action_pred)
                obs = robot_env.get_obs_full()

            self.cloud_array.append(self.process_cloud(obs))
            self.color_array.append(self.process_color(obs))
            # self.depth_array.append(self.process_depth(obs))
            self.env_qpos_array.append(self.process_qpos(obs))
            time.sleep(0.5)
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
    
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
            
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
        return obs_dict
    
    def reset(self):
        if self.debug_mode:
            obs={"qpos": np.zeros(26), "images": {"left": np.zeros((480, 640, 3))}}
        else:
            from robot_env.tianyi_env import tianyi_env as robot_env
            # robot_env.reset_to_prepare_right()
            initial_action = self.process_action(initial_state)
            robot_env.step_full(initial_action)
            # warm up
            time.sleep(2)
            obs = robot_env.get_obs_full()
            # print(obs)
            time.sleep(2)

        agent_pos = np.stack([self.process_qpos(obs)] * self.obs_horizon, axis=0)
        obs_cloud = np.stack([self.process_cloud(obs)] * self.obs_horizon, axis=0)
        obs_img = np.stack([self.process_color(obs)] * self.obs_horizon, axis=0)
        # obs_depth = np.stack([self.process_depth(obs)] * self.obs_horizon, axis=0)

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
        return obs_dict

    def process_action(self, action):
        # TODO: de normalize action
        # left arm position(7), right arm position(7), left hand position(6), right hand position(6)
        left_arm_jpos = action[0:7]
        right_arm_jpos = action[7:14]
        left_hand_jpos = action[14:20]
        right_hand_jpos = action[20:26]
        # Accoroding to inrocs/robot/tianyi_pro4.py, the action is left hand, right hand, left arm, right arm.
        all_actions = np.concatenate((left_hand_jpos, right_hand_jpos, left_arm_jpos, right_arm_jpos), axis=-1)
        return all_actions
    
    def process_cloud(self, obs):
        cam_name = self.camera_names[0]
        rgb_img = obs['images'][cam_name]
        rgb_img = cv2.imdecode(rgb_img, cv2.IMREAD_COLOR)
        depth_img = obs['depths'][cam_name]
        depth_img = cv2.imdecode(depth_img, cv2.IMREAD_COLOR)
        camera_info = obs['color_info']
        camera_intrinsics = np.array(camera_info.K).reshape(3, 3)
        point_cloud = point_cloud_util.create_point_cloud(rgb_img, depth_img, camera_intrinsics)
        obs_extrinsics_matrix = point_cloud_util.get_extrinsics_matrix(camera_info)
        obs_pointcloud = point_cloud_util.preprocess_point_cloud(point_cloud, obs_extrinsics_matrix, use_cuda=True)
        return obs_pointcloud
    
    def process_color(self, obs):
        if len(self.camera_names) != 1:
            raise ValueError("only support one camera")
        cam_name = self.camera_names[0]
        image = obs['images'][cam_name]
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = (image / 255.0).astype(np.float32)
        return image
    
    def process_depth(self, obs):
        return None
    
    def process_qpos(self, obs):
        qpos = obs['qpos']
        left_arm_jpos = qpos[:7]
        left_hand_jpos = qpos[7:13]
        right_arm_jpos = qpos[13:20]
        right_hand_jpos = qpos[20:26]
        arm = np.concatenate([left_arm_jpos, right_arm_jpos])
        hand = np.concatenate([left_hand_jpos, right_hand_jpos])
        qpos = np.concatenate([arm, hand])
        return qpos
    