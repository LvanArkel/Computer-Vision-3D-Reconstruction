import cv2
import json
import numpy as np

camera_names = [f"cam{i}" for i in range(1, 5)]

def load_intrinsic_configs():
    intrinsics = []
    for camera_name in camera_names:
        with open(f"data/{camera_name}/calibration.json") as f:
            calibration = json.load(f)
            intrinsic = {
                "ret": calibration["ret"],
                "mtx": np.array(calibration["mtx"], dtype="float32"),
                "dist": np.array(calibration["dist"], dtype="float32")
            }
            intrinsics.append(intrinsic)
    return intrinsics

def load_configs():
    configs = []
    for camera_name in camera_names:
        with open(f"data/{camera_name}/calibration.json") as f:
            configs.append(json.load(f))
    return configs

def load_vid_in_directory(directory):
    vids = []
    for camera in camera_names:
        vid = cv2.VideoCapture(f"data/{directory}/{camera}.avi")
        vids.append(vid)
    return vids

def load_backgrounds():
    return load_vid_in_directory("background")

def load_extrinsics():
    return load_vid_in_directory("extrinsics")

def load_videos():
    return load_vid_in_directory("video")
