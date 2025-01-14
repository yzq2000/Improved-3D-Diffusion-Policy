# import sys
# sys.path.append('/usr/local/lib/python3.8/dist-packages/')

import os
import rosbag
import numpy as np
from scipy.spatial import cKDTree
import cv2
import h5py

from pathlib import Path
from tqdm import tqdm

class RosbagReader:
    def __init__(self, bag_path: str):
        self.reference_topic = ""
        self.all_aligned_data = {}
        self.bag_path = bag_path
        self.bag = rosbag.Bag(self.bag_path)
        self.topics = list(self.bag.get_type_and_topic_info()[1].keys())
        print(f"Available {len(self.topics)} topics: {self.topics}")
    
    def align_all_msgs(self, reference_topic):
        self.reference_topic = reference_topic
        all_topics_data = {}
        reference_timestamps = []

        for topic, msg, t in self.bag.read_messages():
            if topic not in all_topics_data:
                all_topics_data[topic] = {'timestamps': [], 'messages': []}

            all_topics_data[topic]['timestamps'].append(t.to_sec())
            all_topics_data[topic]['messages'].append(msg)

        reference_timestamps = np.array(all_topics_data[reference_topic]['timestamps'])

        self.all_aligned_data = {reference_topic: {'timestamps': reference_timestamps, 'messages': all_topics_data[reference_topic]['messages']}}

        for topic, data in all_topics_data.items():
            if topic == reference_topic:
                continue

            target_timestamps = np.array(data['timestamps'])
            # find nearest timestamp in reference topic
            tree = cKDTree(target_timestamps.reshape(-1, 1))
            _, indices = tree.query(reference_timestamps.reshape(-1, 1), k=1)

            self.all_aligned_data[topic] = {
                'timestamps': target_timestamps[indices],
                'messages': [data['messages'][i] for i in indices]
            }
        print(f"Data aligned with {reference_topic}, "
              f"total frame count = {len(reference_timestamps)}, "
              f"time duration = {reference_timestamps[-1] - reference_timestamps[0]} s, "
              f"average frequency = {len(reference_timestamps) / (reference_timestamps[-1] - reference_timestamps[0])} Hz")    
    
    def imgmsg_to_numpy(self, camera_message):
        encoding = camera_message.encoding
        width = camera_message.width
        height = camera_message.height
        data = camera_message.data
        if encoding == 'bgr8':
            image_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        elif encoding == 'rgb8':
            image_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        elif encoding == 'mono8':
            image_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        elif encoding == '16UC1':
            image_np = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")
        return image_np
    
    def export_aligned_images(self, topic_name) -> list:
        msgs = self.export_aligned_msgs(topic_name)
        images = []
        for msg in msgs:
            images.append(self.imgmsg_to_numpy(msg))
        return images
    
    def export_aligned_positions(self, topic_name) -> list:
        msgs = self.export_aligned_msgs(topic_name)
        positions = []
        for msg in msgs:
            positions.append(msg.position)
        return positions
    
    def export_aligned_msgs(self, topic_name) -> list:
        if self.reference_topic == "":
            print("No reference topic selected")
            return []
        return self.all_aligned_data[topic_name]['messages']
        
    def export_messages(self, topic_name) -> list:
        messages = []
        for topic, msg, t in self.bag.read_messages(topics=[topic_name]):
            messages.append({
                'timestamp': t.to_sec(),
                'message': msg
            })
        return messages
                
if __name__ == '__main__':
    reader = RosbagReader('/home/a/Projects/3D-Diffusion-Policy/3D-Diffusion-Policy/data_loader/test_bag_2024-12-20-15-47-10.bag')
    reference_topic = '/camera/color/image_raw'
    reader.align_all_msgs(reference_topic)
    rgb_image_raw = reader.export_aligned_msgs("/camera/color/image_raw")
    rgb_camera_info = reader.export_aligned_msgs("/camera/color/camera_info")
    depth_image_raw = reader.export_aligned_msgs("/camera/depth/image_raw")
    depth_camera_info = reader.export_aligned_msgs("/camera/depth/camera_info")
    left_arm_ctrl = reader.export_aligned_msgs("/human_arm_ctrl_left")
    right_arm_ctrl = reader.export_aligned_msgs("/human_arm_ctrl_right")
    left_hand_ctrl = reader.export_aligned_msgs("/inspire_hand/ctrl/left_hand")
    right_hand_ctrl = reader.export_aligned_msgs("/inspire_hand/ctrl/right_hand")
    left_arm_state = reader.export_aligned_msgs("/human_arm_state_left")
    right_arm_state = reader.export_aligned_msgs("/human_arm_state_right")
    left_hand_state = reader.export_aligned_msgs("/inspire_hand/state/left_hand")
    right_hand_state = reader.export_aligned_msgs("/inspire_hand/state/right_hand")
    print("\n-------------rgb_camera_info----------------\n", "rgb_camera_info", len(rgb_camera_info), "\n", rgb_camera_info[0])
    print("\n-------------rgb_image_raw----------------\n", "rgb_image_raw", len(rgb_image_raw), "\n", rgb_image_raw[0])
    print("\n-------------depth_camera_info----------------\n", "depth_camera_info", len(depth_camera_info), "\n", depth_camera_info[0])
    print("\n-------------depth_image_raw----------------\n", "image_raw", len(depth_image_raw), "\n", depth_image_raw[0])
    print("\n-------------right_arm_state----------------\n", len(right_arm_state), "\n", right_arm_state[0])
    print("\n-------------right_hand_state----------------\n", len(right_hand_state), "\n", right_hand_state[0])
