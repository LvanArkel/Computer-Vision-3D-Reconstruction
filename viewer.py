import cv2
import numpy as np

import initialiser

frame_skip = 1

def show_frames(frames):
    # Assumes len(frames)==4
    big_frame = np.concatenate(
        [np.concatenate([frames[0], frames[1]], axis=1),
        np.concatenate([frames[2], frames[3]], axis=1)],
        axis=0
    )
    cv2.imshow("image", big_frame)

if __name__ == "__main__":
    videos = initialiser.load_videos()
    min_frames = min([vid.get(cv2.CAP_PROP_FRAME_COUNT) for vid in videos])
    for i in range(0, int(min_frames), frame_skip):
        frames = [vid.read()[1] for vid in videos]
        show_frames(modify_frames(frames))
        cv2.waitKey(1)