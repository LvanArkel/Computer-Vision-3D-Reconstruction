import cv2
import glm
import random
import numpy as np

import initialiser
import voxels

block_size = 1.0
frame_select = 20

color_map = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 0.0)}


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def initial_voxel_frame(width, height, depth):
    shape = (
        (-width / 2, width / 2),
        (0, height),
        (-depth / 2, depth / 2)
    )
    configs = initialiser.load_configs()
    names = initialiser.camera_names
    frames = [vid.read()[1] for vid in initialiser.load_videos()]
    lookup_table, points = voxels.lookup_table((configs, names), shape, frames[0].shape)
    print("Generated lookup table")
    # TODO: Get masks for first frames
    masks = [np.ones(frames[0].shape[:-1]) for i in range(len(frames))]
    print(masks[0].shape)
    active_voxels, active_colors, _ = voxels.get_colored_voxel_model(lookup_table, points, names, frames, masks, 0)
    return active_voxels, active_colors


def voxel_model_animation(width, height, depth):
    shape = (
        (-width / 2, width / 2),
        (0, height),
        (-depth / 2, depth / 2)
    )
    print("Starting voxel model generation")
    configs = initialiser.load_configs()
    names = initialiser.camera_names
    videos = initialiser.load_videos()
    frame_width = videos[0].get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_shape = frame_width, frame_height
    # Initialise lookup table
    lookup_table, points = voxels.lookup_table((configs, names), shape, frame_shape)
    print("Generated lookup table")
    # Go through each frame
    min_frames = int(min([vid.get(cv2.CAP_PROP_FRAME_COUNT) for vid in videos]))
    for frame_i in range(0, min_frames, frame_select):
        frames = []
        for vid in videos:
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
            ret, frame = vid.read()
            if not ret:
                return
            frames.append(frame)
        # TODO: Background subtraction to get masks
        masks = [np.ones(frames[0].shape[:-1]) for i in range(len(frames))]
        # Retrieving the active voxels and colors, currently ignoring indices
        active_voxels, active_colors, _ = voxels.get_colored_voxel_model(lookup_table, points, names, frames, masks, 0)
        yield active_voxels, active_colors


def set_voxel_positions(width, height, depth):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    for active_voxels, active_colors in voxel_model_animation(width, height, depth):
        # yield active_voxels, active_colors/255
        # continue
        ret, labels, centers = cv2.kmeans(active_voxels, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        print("Generating frame")
        colored_voxels = [color_map[label[0]] for label in labels]
        yield active_voxels, colored_voxels, centers[:, [0, 2]]


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
           [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
