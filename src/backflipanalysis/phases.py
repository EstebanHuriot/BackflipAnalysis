import pandas as pd
import numpy as np

def detect_set(df_knee, min_flexion=30, initial_frames=10):
    """
    Detects the end of the set phase, defined as first significant local minima of the knee angle (max knee flexion)
    Input:
        - df_knee: a pd.df containing frame, time and knee angle
        - min_flexion: arbitraty minimum flexion to consider a local minimum significant
        - initial_frames: number of frames used to estimate the standing knee angle
    Outputs: int or None, frame corresponding to the end of the set
    
    """
    # estimates standing knee angle
    initial_angle = df_knee['angle'].iloc[:initial_frames].median()

    # °/s
    velocity = np.gradient(df_knee['angle'], df_knee['time'])

    # derivative changes
    candidates = np.where((velocity[:-1] < 0) & (velocity[1:] >= 0))[0] + 1

    # keep the first significant change in knee flexion
    for idx in candidates:

        flexion = initial_angle - df_knee.loc[idx, "angle"]

        if flexion >= min_flexion:
            return int(df_knee.loc[idx, "frame"])

    return None