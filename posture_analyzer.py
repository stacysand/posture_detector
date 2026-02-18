SHOULDER_TILT_THRESHOLD = 0.03  # normalized by frame height; tune this value

def analyze_shoulders(left_shoulder_px, right_shoulder_px, frame_height):
    """
    Checks if shoulders are level.

    Args:
        left_shoulder_px:  (x, y) pixel coords of left shoulder
        right_shoulder_px: (x, y) pixel coords of right shoulder
        frame_height:      height of the frame in pixels (for normalization)

    Returns:
        dict with:
            'bad_posture' (bool)   — True if shoulders are tilted
            'tilt'        (float)  — normalized y-difference (0.0–1.0)
    """
    left_y  = left_shoulder_px[1]
    right_y = right_shoulder_px[1]

    tilt = abs(left_y - right_y) / frame_height  # normalize

    return {
        'bad_posture': tilt > SHOULDER_TILT_THRESHOLD,
        'tilt': tilt
    }
