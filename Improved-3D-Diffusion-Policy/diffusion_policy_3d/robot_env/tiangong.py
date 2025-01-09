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
import tqdm
import torch
import os 
import cv2
import numpy as np
import sys


# Add the project directory to the Python path
project_path = str(pathlib.Path(__file__).parent.parent)
sys.path.append(project_path)


sys.path.append('/home/ps/Dev/vibrant/inrocs_1129_pi0/inrocs')
from robot_env.tianyi_env import tianyi_env as robot_env

class TiangongDexEnvInference:
    """
    The deployment is running on the local computer of the robot.
    """
    def __init__(self, obs_horizon=2, action_horizon=8, device="gpu",
                use_point_cloud=True, use_image=True, img_size=84,
                 num_points=1024,camera_names=['top']):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.img_size=img_size
        self.num_points=num_points
        self.camera_names=camera_names
        
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        # init robot env
        robot_env.reset_to_prepare_left()
    
    def step(self, action_list):
        
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            
            # de normalize action
            action_pred = self.process_action(act)
            
            # debug
            # robot_env.stepfull(action_pred)

            obs = robot_env.get_obs_full()
            
            self.cloud_array.append(self.process_cloud(obs))
            self.color_array.append(self.process_color(obs))
            self.depth_array.append(self.process_depth(obs))
            self.env_qpos_array.append(self.process_qpos(obs))
            time.sleep(0.2)
        
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
    
    def reset(self, first_init=True):
        robot_env.reset_to_prepare_left()
        # warm up
        import time
        time.sleep(2)
        obs = robot_env.get_obs()
        agent_pos = self.process_qpos(obs)
        obs_cloud = self.process_cloud(obs)
        obs_img = self.process_color(obs)
        obs_depth = self.process_depth(obs)
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
        return obs_dict

    # de normalize action
    def process_action(self, action):
        # TODO: de normalize action
        # left arm position(7), right arm position(7), left hand position(6), right hand position(6)
        left_arm_jpos = action[:, :, 0:7]
        right_arm_jpos = action[:, :, 7:14]
        left_hand_jpos = action[:, :, 14:20]
        right_hand_jpos = action[:, :, 20:26]
        all_actions = np.concatenate((left_arm_jpos, left_hand_jpos, right_arm_jpos, right_hand_jpos), axis=-1)
        return action
    
    def process_cloud(self, obs):
        # TODO: mathod in data converter
        return None
    
    def process_color(self, obs, show_img=True):
        # (w, h) for cv2.resize
        # print("debug", obs.keys(), obs['images'].keys(), obs['images'])
        img_new_size = (self.img_size, self.img_size) #(480, 640)
        all_cam_images = []
        for cam_name in self.camera_names:
            curr_image = obs['images'][cam_name]
            # print("debug", curr_image)
            curr_image = cv2.imdecode(curr_image, cv2.IMREAD_COLOR)
            curr_image = cv2.resize(curr_image, dsize=img_new_size)
            if show_img:
                rgb_image = curr_image[:, :, ::-1]
                cv2.imshow(f"{cam_name} image", rgb_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  
            all_cam_images.append(curr_image)
        all_cam_images = (all_cam_images / 255.0).astype(np.float32)
        return all_cam_images
    
    def process_depth(self, obs):
        return None
    
    def process_qpos(self, obs):
        qpos = obs['qpos']
        left_jpos = qpos[:7]
        left_hand_jpos = qpos[7:13]
        right_jpos = qpos[13:20]
        right_hand_jpos = qpos[20:26]
        qpos = np.concatenate((left_hand_jpos, right_hand_jpos, left_jpos, right_jpos))
        return qpos
    