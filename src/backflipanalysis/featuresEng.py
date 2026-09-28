import numpy as np

def calculate_angle(a, b, c):

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