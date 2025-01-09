import sys
import os
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from diffusion_policy_3d.robot_env.tiangong import TiangongDexEnvInference
import diffusion_policy_3d.common.gr1_action_util as action_util
import diffusion_policy_3d.common.rotation_util as rotation_util
import tqdm
import torch
import os 
os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


import numpy as np
import torch
from termcolor import cprint

# Add the project directory to the Python path
project_path = str(pathlib.Path(__file__).parent.parent)
sys.path.append(project_path)

@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
)

def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ == 'DPWorkspace':
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True
        
    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 1000,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    
    img_size = 84
    num_points = 1024
    first_init = True
    record_data = True

    env = TiangongDexEnvInference(obs_horizon=2, action_horizon=action_horizon, device="cpu",
                                use_point_cloud=use_point_cloud,
                                use_image=use_image,
                                img_size=img_size,
                                num_points=num_points,
                                camera_names=['top'])

    
    obs_dict = env.reset(first_init=first_init)

    step_count = 0
    
    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[0]
            action_list = [act.numpy() for act in action]
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step: {step_count}")

    if record_data:
        import h5py
        root_dir = "/home/ps/saved_record_data/"
        save_dir = root_dir + "deploy_dir"
        os.makedirs(save_dir, exist_ok=True)
        
        record_file_name = f"{save_dir}/demo.h5"
        color_array = np.array(env.color_array)
        depth_array = np.array(env.depth_array)
        cloud_array = np.array(env.cloud_array)
        qpos_array = np.array(env.qpos_array)
        with h5py.File(record_file_name, "w") as f:
            f.create_dataset("color", data=np.array(color_array))
            f.create_dataset("depth", data=np.array(depth_array))
            f.create_dataset("cloud", data=np.array(cloud_array))
            f.create_dataset("qpos", data=np.array(qpos_array))
        
        choice = input("whether to rename: y/n")
        if choice == "y":
            renamed = input("file rename:")
            os.rename(src=record_file_name, dst=record_file_name.replace("demo.h5", renamed+'.h5'))
            new_name = record_file_name.replace("demo.h5", renamed+'.h5')
            cprint(f"save data at step: {roll_out_length} in {new_name}", "yellow")
        else:
            cprint(f"save data at step: {roll_out_length} in {record_file_name}", "yellow")


if __name__ == "__main__":
    main()
