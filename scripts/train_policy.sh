# Examples:

# bash scripts/train_policy.sh dp_224x224_r3m  tiangong_dexhand-image 0113-image-skin-100 tiangong_dexhand_skin_100.zarr
# bash scripts/train_policy.sh idp3  tiangong_dexhand-3d 0113-3d-skin-100 tiangong_dexhand_skin_100.zarr
# bash scripts/train_policy.sh dp_224x224_r3m  tiangong_dexhand-image 0113-image-grasp-82 tiangong_dexhand_grasp_82.zarr
# bash scripts/train_policy.sh idp3  tiangong_dexhand-3d 0113-3d-grasp-82 tiangong_dexhand_grasp_82.zarr

DEBUG=False
wandb_mode=online


alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
dataset_name=${4}

seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    save_ckpt=False
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    save_ckpt=True
    echo -e "\033[33mTrain mode\033[0m"
fi


cd Improved-3D-Diffusion-Policy

dataset_path="$(pwd)/real_data/${dataset_name}"
# echo $dataset_path

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 



                                