import numpy as np
import pandas as pd
import cv2

def get_video_properties(video_path):
    """ 
    Extract video resolution and frame rate.
    Takes video path as input in str
    Outputs a tuple with video width, height and FPS
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return width, height, fps



def calculate_angle(a, b, c):
    """ 
    Calculate 2D angle ABC in degrees
    Input a, b, c (each is np.array)
    ! Keep in mibd that b is the angle's vertex !
    Output is an angle in degrees
    """

    # Both vectors start from B
    ba = a - b
    bc = c - b

    # Calculate vector magnitudes
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    # Cosine of the angle
    re = np.dot(ba, bc) / (norm_ba * norm_bc)

    # Convert cosine to angle in degrees
    angle = np.degrees(np.arccos(np.clip(re, -1, 1)))

    return angle



def calculate_angle_series(df, a, b , c, width, height, fps):
    """
    calcule_angle but for every frame in a video
    Input df (pd.df) containing frame, keypoint, x and y ,
    a, b , c like in calculate_angle
    width, height, fps as resolution and frame rate
    Output a df with frame index, timestamp and angle in degree
    """
    angles = []

    for frame_idx, frame in df.groupby("frame"):

        frame = frame.set_index("keypoint")

        point_a = frame.loc[a]
        point_b = frame.loc[b]
        point_c = frame.loc[c]

        scale = np.array([width, height])

        a2 = point_a[["x", "y"]].to_numpy(dtype=float) * scale
        b2 = point_b[["x", "y"]].to_numpy(dtype=float) * scale
        c2 = point_c[["x", "y"]].to_numpy(dtype=float) * scale

        angle = calculate_angle(a2, b2, c2)

        angles.append({"frame": frame_idx, "time": frame_idx / fps,"angle": angle})

    return pd.DataFrame(angles)