from collections import defaultdict
import os

import initialiser
import cv2 as cv
import numpy as np

voxel_size = 2


def create_lookup_table(config_names, voxel_shape, frame_shape):
    configs, names = config_names
    x_shape, y_shape, z_shape = voxel_shape
    # Note: Maybe move around center of voxel model. Perhaps checking the background model again
    points = np.mgrid[
             x_shape[0]:x_shape[1],
             y_shape[0]:y_shape[1],
             z_shape[0]:z_shape[1]
             ].T.reshape(-1, 3)
    lookup_table = dict()
    for config, name in zip(configs, names):
        lookup = defaultdict(set)
        mtx = np.array(config["mtx"], dtype="float32")
        dist = np.array(config["dist"], dtype="float32")
        cam_rot = np.array(config["rvecs"], dtype="float32")
        tvecs = np.array(config["tvecs"], dtype="float32")
        imgpoints, _ = cv.projectPoints((points * voxel_size).astype("float32"), cam_rot, tvecs, mtx, dist)
        # origin, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype="float32"), cam_rot, tvecs, mtx, dist)
        # print(f"{name} origin at {origin}")
        # print(name, imgpoints)
        for point_i, img in enumerate(imgpoints):
            if not in_bounds(frame_shape, img[0]):
                continue
            imgpoint = (int(img[0, 0]), int(img[0, 1]))
            lookup[imgpoint].add(point_i)
            # if imgpoint not in lookup.keys():
            #     lookup[imgpoint] = {point_i}
            # else:
            #     lookup[imgpoint].add(point_i)
        lookup_table[name] = lookup
    return lookup_table, points


def lookup_table(config_names, voxel_shape, frame_shape):
    # TODO: Find saved lookup table
    cached = initialiser.load_lookup_table(voxel_shape)
    if cached is None:
        cached = create_lookup_table(config_names, voxel_shape, frame_shape)
        initialiser.save_lookup_table(cached, voxel_shape)
    return cached


def in_bounds(frame_shape, key):
    return 0 <= key[1] < frame_shape[0] and 0 <= key[0] < frame_shape[1]


def get_voxels_for_camera(lookup_table, name, frame, mask):
    """
    Retrieves all voxels that are enabled for a mask given a camera name
    :param lookup_table: The lookup table containing all data
    :param name: The name of the camera
    :param frame: The colored frame of the image
    :param mask: A binary image containing the foreground
    :return: A list of pairs of voxels in the form of (voxels, color)
    """
    lookup = lookup_table[name]
    colored_voxels = [(voxel, frame[key[1], key[0]]) for key, voxels in lookup.items() if mask[key[1], key[0]] == 1
                      for voxel in voxels]
    return colored_voxels


def cluster_voxels(voxels):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(voxels, 4, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    return label, center


def get_colored_voxel_model(lookup_table, points, names, frames, masks, color_camera):
    camera_colored_voxels = [get_voxels_for_camera(lookup_table, name, frame, mask) for
                             name, frame, mask in zip(names, frames, masks)]
    vox_to_col = dict(camera_colored_voxels[color_camera])
    voxels_per_camera = [{voxel for voxel, color in colored_voxels} for colored_voxels in camera_colored_voxels]
    active_voxel_indices = set.intersection(*voxels_per_camera)
    voxels = np.array([points[vox_i] for vox_i in active_voxel_indices], dtype="float32")
    colors = np.array([vox_to_col[vox_i] for vox_i in active_voxel_indices])
    return voxels, colors, active_voxel_indices


def get_active_voxels(lookup_table, points, names, frames, masks):
    camera_colored_voxels = [get_voxels_for_camera(lookup_table, name, frame, mask) for
                             name, frame, mask in zip(names, frames, masks)]
    voxels_per_camera = [{voxel for voxel, color in colored_voxels} for colored_voxels in camera_colored_voxels]
    active_voxels = set.intersection(*voxels_per_camera)
    voxels = np.array([points[vox_i] for vox_i in active_voxels], dtype="float32")
    labels, centers = cluster_voxels(voxels)
    return voxels, labels, centers
