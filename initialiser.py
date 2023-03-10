import os.path

import cv2
import json
import pickle
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
                "dist": np.array(calibration["dist"], dtype="float32"),
            }
            intrinsics.append(intrinsic)
    return intrinsics

def load_configs():
    configs = []
    for camera_name in camera_names:
        with open(f"data/{camera_name}/config.json") as f:
            config = json.load(f)
            config["mtx"] = np.array(config["mtx"], dtype="float32")
            config["dist"] = np.array(config["dist"], dtype="float32")
            config["rvecs"] = np.array(config["rvecs"], dtype="float32")
            config["tvecs"] = np.array(config["tvecs"], dtype="float32")
            configs.append(config)
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


def load_lookup_table(voxel_shape):
    if os.path.exists("lookup.json"):
        with open("lookup.json", "rb") as f:
            contents = pickle.load(f)
            if "shape" in contents.keys() and contents["shape"] == voxel_shape:
                lookup_table = contents["lookup_table"]
                points = np.array(contents["points"], dtype="float32")
                return lookup_table, points
    return None


def save_lookup_table(cached, voxel_shape):
    with open("lookup.json", "wb") as f:
        lookup_table, points = cached
        obj = {
            "shape": voxel_shape,
            "lookup_table": lookup_table, # {k1: {f"{k2[0]}#{k2[1]}": list(v2) for k2, v2 in v1.items()} for k1, v1 in lookup_table.items()},
            "points": points.tolist()
        }
        pickle.dump(obj, f)
