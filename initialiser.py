import cv2

camera_names = [f"cam{i}" for i in range(1, 5)]

def load_intrinsics():
    intrinsics = []
    for camera_name in camera_names:
        with open(f"data/{camera_name}/calibration.json"):
            pass

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
