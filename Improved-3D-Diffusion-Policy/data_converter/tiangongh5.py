
import sys
sys.path.append('/usr/local/lib/python3.8/dist-packages/')

import os
import rosbag
import numpy as np
from scipy.spatial import cKDTree
import cv2
import h5py

from pathlib import Path
from tqdm import tqdm


rosbag_root_path = "/nfsroot/DATA/users/embodied_ai/vibrant/test"
h5_root_path = "/nfsroot/DATA/users/embodied_ai/vibrant/test/h5"


def imgmsg_to_numpy(camera_message):
    # 获取图像的编码格式
    encoding = camera_message.encoding

    # 获取图像的尺寸和字节数据
    width = camera_message.width
    height = camera_message.height
    data = camera_message.data

    # 根据不同的编码格式，解码数据
    if encoding == 'bgr8':
        # 三通道，BGR 格式
        image_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    elif encoding == 'rgb8':
        # 三通道，RGB 格式
        image_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    elif encoding == 'mono8':
        # 单通道，灰度图像
        image_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
    elif encoding == '16UC1':
        # 单通道，16位无符号整数
        image_np = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
    else:
        # 当遇到未知编码格式时，抛出一个错误
        raise ValueError(f"Unsupported encoding: {encoding}")

    return image_np
def set_h5_data( np_data, h5_file_path,set_compress):
        camera_color_images = np_data['/camera/color/image_raw']
        camera_depth_images =  np_data['/camera/depth/image_raw']
        aligned_left_joint_data = np_data['/human_arm_ctrl_left']
        aligned_right_joint_data = np_data['/human_arm_ctrl_right']
        aligned_left_end_data = np_data['/inspire_hand/ctrl/left_hand']
        aligned_right_end_data = np_data['/inspire_hand/ctrl/right_hand']

        camera_rgb_image_all = []
        camera_depth_image_all = []
        # print('camera_color_images:',camera_color_images.shape)
        # print('camera_depth_images:',camera_depth_images.shape)
        for i in range(len(camera_color_images)):
            camera_rgb_image = camera_color_images[i]
            camera_depth_image = camera_depth_images[i]

            # np.save('camera_front_depth_image_0.npy', camera_front_depth_image)
            # quit()
            if set_compress:
                camera_rgb_image = cv2.imencode(".jpg", camera_rgb_image)[1]#.tobytes()
                camera_depth_image = cv2.imencode(".png", camera_depth_image)[1]#.tobytes()

            camera_rgb_image_all.append(camera_rgb_image)
            camera_depth_image_all.append(camera_depth_image)

        if not set_compress:
            camera_rgb_image_all = np.asarray(camera_rgb_image_all)
            camera_depth_image_all = np.asarray(camera_depth_image_all)

        # print('aligned_left_joint_data:',aligned_left_joint_data.shape)
        # print('aligned_right_joint_data:',aligned_right_joint_data.shape)

        # print('aligned_left_end_data:',aligned_left_end_data.shape)
        # print('aligned_right_end_data:',aligned_right_end_data.shape)

        joint_position_all = np.c_[aligned_left_joint_data, aligned_right_joint_data]
        end_effector_all = np.c_[aligned_left_end_data, aligned_right_end_data]

        with h5py.File(h5_file_path, 'w') as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = set_compress
            obs = root.create_group('observations')
            rgb_images = obs.create_group('rgb_images')
            depth_images = obs.create_group('depth_images')

            if set_compress:
                dt = h5py.vlen_dtype(np.dtype('uint8'))

                dset_top_images = rgb_images.create_dataset('camera_top',
                                                            (len(camera_rgb_image_all),), dtype=dt)
                for i, camera_top_rgb_image in enumerate(camera_rgb_image_all):
                    dset_top_images[i] = camera_top_rgb_image.flatten()

                ############################
                dt2 = h5py.special_dtype(vlen=np.dtype('uint8'))

                dset_top_depth = depth_images.create_dataset('camera_top',
                                                             (len(camera_depth_image_all),), dtype=dt2)
                for i, camera_top_depth_image in enumerate(camera_depth_image_all):
                    dset_top_depth[i] = np.frombuffer(camera_top_depth_image.tobytes(), dtype=np.uint8)

            else:
                rgb_images.create_dataset('camera_top', data=camera_rgb_image_all)
                depth_images.create_dataset('camera_top', data=camera_depth_image_all)

            master = root.create_group('master')
            master.create_dataset('joint_position', data=joint_position_all)
            master.create_dataset('end_effector', data=end_effector_all)
            puppet = root.create_group('puppet')
            puppet.create_dataset('joint_position', data=joint_position_all)
            puppet.create_dataset('end_effector', data=end_effector_all)
def find_nearest_indices_np(reference_timestamps, target_timestamps):
    """找到与参考时间戳最近的目标时间戳的索引，使用NumPy向量化操作。

    参数:
    - reference_timestamps: 一个数组，包含参考时间戳。
    - target_timestamps: 一个数组，包含目标时间戳。

    返回:
    - 一个数组，包含与每个参考时间戳最近的目标时间戳的索引。
    """
    indices = []
    for ref_ts in reference_timestamps:
        # 计算每个参考时间戳与目标时间戳之间的绝对差值
        differences = np.abs(target_timestamps - ref_ts)
        # 找到最小差值的位置索引
        nearest_index = np.argmin(differences)
        indices.append(nearest_index)
    return np.array(indices)

def find_nearest_indices(reference_timestamps, target_timestamps):
    """找到与参考时间戳最近的目标时间戳的索引。"""
    tree = cKDTree(target_timestamps.reshape(-1, 1))
    _, indices = tree.query(reference_timestamps.reshape(-1, 1), k=1)
    return indices

def extract_and_align_data(bag_file_path, reference_topic):
    bag = rosbag.Bag(bag_file_path)

    # 用于存储每个 topic 的时间戳和数据
    all_topics_data = {}
    reference_timestamps = []

    # 首先遍历所有消息收集它们的时间戳和数据
    for topic, msg, t in bag.read_messages():
        if topic not in all_topics_data:
            all_topics_data[topic] = {'timestamps': [], 'messages': []}

        all_topics_data[topic]['timestamps'].append(t.to_sec())
        all_topics_data[topic]['messages'].append(msg)

    # 提取参考 topic 的时间戳
    reference_timestamps = np.array(all_topics_data[reference_topic]['timestamps'])

    # 创建字典存储对齐后的数据
    aligned_data = {reference_topic: {'timestamps': reference_timestamps, 'messages': all_topics_data[reference_topic]['messages']}}

    # 处理其他 topics
    for topic, data in all_topics_data.items():
        if topic == reference_topic:
            continue

        target_timestamps = np.array(data['timestamps'])
        indices = find_nearest_indices(reference_timestamps, target_timestamps)
        # indices_np =  find_nearest_indices_np(reference_timestamps, target_timestamps)

        aligned_data[topic] = {
            'timestamps': target_timestamps[indices],
            'messages': [data['messages'][i] for i in indices]
        }

    bag.close()
    return aligned_data

def process_bag(bag):
    reference_topic = '/camera/color/image_raw'
    try:
        bag_file_path = str(bag)

        # 提取和对齐数据
        aligned_data = extract_and_align_data(bag_file_path, reference_topic)

        # 示例打印对齐后的数据字典
        # print(f"Aligned data for bag: {bag_file_path}")
        # for topic in aligned_data:
            # print(f"  Topic: {topic}, Number of messages: {len(aligned_data[topic]['messages'])}")
        data_keys = ['/camera/color/camera_info', '/human_arm_state_right', '/human_arm_state_left', 
                        '/human_arm_ctrl_left', '/human_arm_ctrl_right', '/inspire_hand/state/right_hand', 
                        '/inspire_hand/state/left_hand', '/human_arm_6dof_right', '/inspire_hand/ctrl/left_hand', 
                        '/inspire_hand/ctrl/right_hand', '/camera/depth/camera_info', '/camera/depth/image_raw', 
                        '/camera/color/image_raw']


        np_data = {}
        # print("all keys", len(data_keys), data_keys)
        for key in data_keys:
            time_diff =  (aligned_data[key]['timestamps'] -  aligned_data['/camera/color/image_raw']['timestamps'])*1000
            # print(key,":ms","mean:",time_diff.mean(),"max:",time_diff.max(),"min:",time_diff.min())
            data_list = []
            # print("timestamps", len(aligned_data[key]['timestamps']))
            if key in ['/camera/color/image_raw','/camera/depth/image_raw']:

                for image in aligned_data[key]['messages']:
                    img_np = imgmsg_to_numpy(image)
                    data_list.append(img_np)   
                np_data[key] = np.array(data_list)[:200]
            elif key in [ '/human_arm_state_right','/human_arm_state_left','/human_arm_ctrl_left','/human_arm_ctrl_right','/inspire_hand/state/right_hand','/inspire_hand/state/left_hand','/inspire_hand/ctrl/left_hand', '/inspire_hand/ctrl/right_hand']:
                for data in aligned_data[key]['messages']:
                    data_list.append(data.position)
                np_data[key] = np.array(data_list) [:200]
            elif key in ['/camera/color/camera_info', '/camera/depth/camera_info']:
                # for data in aligned_data[key]['messages']:
                #     data_list.append(data.position)
                # np_data[key] = np.array(data_list) [:200]
                print("debug", len(aligned_data[key]['messages']), key, aligned_data[key]['messages'])
        h5_dir_name = bag_file_path[-23: -4]
        h5_path = os.path.join(h5_root_path, h5_dir_name)
        # print("debug", h5_path)
        if not os.path.exists(h5_path):
            os.makedirs(h5_path)
        file_path = os.path.join(h5_path,"trajectory.hdf5")
        print(len(np_data.keys()), np_data.keys())
        # for key, value in np_data.items():
            # print(key, value.shape, len(value))
            # print("\n\n")
        set_h5_data(np_data,file_path,True)
    except Exception as e:
        print(e)
        pass

def main(directory,h5_path,reference_topic):
    # directory = "/home/leimao/data/tiangong_mt"
    # reference_topic = "/camera/color/camera_info"
    # reference_topic = "/camera/color/image_raw"

    datasets = Path(directory)

    bags = list(datasets.rglob('test_bag*.bag'))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=20) as executor:
            list(tqdm(executor.map(process_bag, bags), total=len(bags)))

    # for bag in tqdm(bags):

    # for root, dirs, files in os.walk(directory):
    #     for file in files:
    #         if file.endswith(".bag"):
    #             bag_file_path = os.path.join(root, file)

    #             # 提取和对齐数据
    #             aligned_data = extract_and_align_data(bag_file_path, reference_topic)

    #             # 示例打印对齐后的数据字典
    #             print(f"Aligned data for bag: {bag_file_path}")
    #             for topic in aligned_data:
    #                 print(f"  Topic: {topic}, Number of messages: {len(aligned_data[topic]['messages'])}")
    #             data_keys = ['/camera/color/camera_info', '/human_arm_state_right', '/human_arm_state_left', 
    #                           '/human_arm_ctrl_left', '/human_arm_ctrl_right', '/inspire_hand/state/right_hand', 
    #                           '/inspire_hand/state/left_hand', '/human_arm_6dof_right', '/inspire_hand/ctrl/left_hand', 
    #                           '/inspire_hand/ctrl/right_hand', '/camera/depth/camera_info', '/camera/depth/image_raw', 
    #                           '/camera/color/image_raw']


    #             np_data = {}
    #             for key in data_keys:
    #                 time_diff =  (aligned_data[key]['timestamps'] -  aligned_data['/camera/color/image_raw']['timestamps'])*1000
    #                 print(key,":ms","mean:",time_diff.mean(),"max:",time_diff.max(),"min:",time_diff.min())
    #                 data_list = []
    #                 if key in ['/camera/color/image_raw','/camera/depth/image_raw']:

    #                     for image in aligned_data[key]['messages']:
    #                         img_np = imgmsg_to_numpy(image)
    #                         data_list.append(img_np)   
    #                     np_data[key] = np.array(data_list)
    #                 elif key in [ '/human_arm_state_right','/human_arm_state_left','/human_arm_ctrl_left','/human_arm_ctrl_right','/inspire_hand/state/right_hand','/inspire_hand/state/left_hand','/inspire_hand/ctrl/left_hand', '/inspire_hand/ctrl/right_hand']:
    #                     for data in aligned_data[key]['messages']:
    #                         data_list.append(data.position)
    #                     np_data[key] = np.array(data_list) 
    #             h5_path = os.path.join(h5_path,file[0:-4],"data")
    #             if not os.path.exists(h5_path):
    #                 os.makedirs(h5_path)
    #             file_path = os.path.join(h5_path,"trajectory.hdf5")
    #             set_h5_data(np_data,file_path,True)


if __name__ == "__main__":
    reference_topic = "/camera/color/image_raw"   # topic to align all the data
    main(rosbag_root_path,h5_root_path,reference_topic)