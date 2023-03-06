import cv2
import json
import numpy as np
import os

import initialiser

cols = 8
rows = 6
cell_size = 11.5 # In cm
font = cv2.FONT_HERSHEY_SIMPLEX
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = cell_size * np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def interpolate(points):
    p1, p2, p3, p4 = points
    points = []
    for r in range(rows):
        for c in range(cols):
            alpha = r / (rows-1)
            beta = c / (cols-1)
            point = (1-alpha)*((1-beta)*p1+beta*p2)\
                + alpha * ((1-beta)*p3+beta*p4)
            points.append([np.array([point[0], point[1]], dtype="float32")])
    return np.array(points)

def get_points(frame):
    points = []
    img = frame.copy()
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append(np.array([x, y]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(img, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
            # Add text if p4
            if len(points) >= 4:
                cv2.putText(img, "Enough points found, press any key", (0, 30), font, 0.8, (255, 0, 0), 2)
            cv2.imshow("image", img)
    cv2.imshow("image", img)
    cv2.setMouseCallback('image', click_event)
    while len(points) < 4:
        key = cv2.waitKey(0)
    return points

def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 1)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 1)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 1)
    return img

def find_extriniscs_of_frame(frame, mtx, dist):
    outer_corners = np.array(get_points(frame), dtype="float32")
    warped_corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]], dtype="float32") * 80
    M = cv2.getPerspectiveTransform(outer_corners, warped_corners)
    warped_frame = cv2.warpPerspective(frame, M, (cols * 80, rows * 80))
    inner_corners = np.array(get_points(warped_frame), dtype="float32")
    chess_corners = interpolate(inner_corners)
    corners = cv2.perspectiveTransform(chess_corners, np.linalg.inv(M))

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)*11
    rotated_objp = objp.copy()
    rotated_objp[:, [1, 2]] = rotated_objp[:, [2, 1]]

    ret, rvecs, tvecs = cv2.solvePnP(rotated_objp, corners, mtx, dist)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw_axis(frame, corners, imgpts)
    cv2.putText(img, f"{rvecs}", (0, 30), font, 0.8, (255, 0, 0), 2)
    cv2.putText(img, f"{tvecs}", (0, 60), font, 0.8, (255, 0, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    return rvecs, tvecs


def calibrate(recalculate = False):
    names = initialiser.camera_names
    intrinsics = initialiser.load_intrinsic_configs()
    extrinsic_vids = initialiser.load_extrinsics()
    for name, intrinsic_cfg, video in zip(names, intrinsics, extrinsic_vids):
        if os.path.exists(f"data/{name}/config.json") and not recalculate:
            continue
        mtx, dist = intrinsic_cfg["mtx"], intrinsic_cfg["dist"]
        video.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame = video.read()
        if not ret:
            raise RuntimeError(f"Error, extrinsic video for {name} not loaded")
        rvecs, tvecs = find_extriniscs_of_frame(frame, mtx, dist)
        intrinsic_cfg["rvecs"] = rvecs
        intrinsic_cfg["tvecs"] = tvecs
        print(type(rvecs))
        print(type(tvecs))
        print("Config for camera", name)
        print(intrinsic_cfg)
        with open(f"data/{name}/config.json", "w+") as f:
            json.dump(intrinsic_cfg, f, cls=NumpyEncoder)

if __name__ == "__main__":
    calibrate(recalculate=True)