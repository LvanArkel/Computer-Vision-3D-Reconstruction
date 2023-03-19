import cv2
import glm
import random
import numpy as np

import initialiser
import voxels

from offline import get_background_model, find_camera_foreground, hist, compare, compare2, gaussian, compare_gaussian

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


def offline(active_voxels, active_colors, labels):

    people= [[],[],[],[]]
    for active_voxel, active_color, label in zip(active_voxels, active_colors, labels):
        if active_voxel[1] > 12:
            if label == 0:
                people[0].append(active_color)
            elif label == 1:
                people[1].append(active_color)
            elif label == 2:
                people[2].append(active_color)
            elif label == 3:
                people[3].append(active_color)
    
    histograms = []
    for person in people:
        histograms.append(hist(person))

    #histograms is a list of 4 elements that each contains 3 histograms
    return histograms

def voxel_model_animation(width, height, depth, configs):
    shape = (
        (-width / 2, width / 2),
        (0, height),
        (-depth / 2, depth / 2)
    )
    print("Starting voxel model generation")
    names = initialiser.camera_names
    videos = initialiser.load_videos()
    backgrounds = initialiser.load_backgrounds()
    b_models = []
    for background in backgrounds:
        b_models.append(get_background_model(background))
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
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if not ret:
                return
            frames.append(frame_hsv)

        masks = []
        for frame, background in zip(frames, b_models):
            masks.append(find_camera_foreground(background, frame))
        #masks = [np.ones(frames[0].shape[:-1]) for i in range(len(frames))]
        # Retrieving the active voxels and colors, currently ignoring indices
        active_voxels, active_colors, _ = voxels.get_colored_voxel_model(lookup_table, points, names, frames, masks)
        yield active_voxels, active_colors

def camera_distances(center_imgpoints):
    xcoords = sorted([center_imgpoints[i,0,0] for i in range(len(center_imgpoints))])
    return min([xcoords[i+1] - xcoords[i] for i in range(len(center_imgpoints)-1)])


def set_voxel_positions(width, height, depth):
    configs = initialiser.load_configs()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    color_model = None
    for active_voxels, active_cam_colors in voxel_model_animation(width, height, depth, configs):
        ret, labels, centers = cv2.kmeans(active_voxels[:, [0, 2]], 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers3d = np.array([[x, 0, y] for x, y in centers])*voxels.voxel_size
        cam_smallest_angle = []
        for config in configs:
            center_imgpoints, _ = cv2.projectPoints(centers3d, config["rvecs"],
                                                 config["tvecs"], config["mtx"], config["dist"])
            cam_smallest_angle.append(camera_distances(center_imgpoints))
        best_camera = np.argmax(cam_smallest_angle)
        active_colors = active_cam_colors[:, best_camera]

        if color_model == None:
            color_model = offline(active_voxels, active_colors, labels)

        #yield active_voxels, active_colors/255, np.zeros((4,2))
        #continue

        print("Generating frame")
        

        people= [[],[],[],[]]
        for active_voxel, active_color, label in zip(active_voxels, active_colors, labels):
            if active_voxel[1] > 12:
                if label == 0:
                    people[0].append(active_color)
                elif label == 1:
                    people[1].append(active_color)
                elif label == 2:
                    people[2].append(active_color)
                elif label == 3:
                    people[3].append(active_color)
            

        histograms = []
        for person in people:
            histograms.append(hist(person))


        matched_labels = compare2(color_model, histograms)

        colored_voxels = [color_map[matched_labels[label[0]]] for label in labels]
        #color_model = histograms
        yield active_voxels, colored_voxels, centers


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
