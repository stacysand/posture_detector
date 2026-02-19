SHOULDER_TILT_THRESHOLD = 0.028  # normalized by frame height; tune this value
HEAD_DISTANCE_THRESHOLD  = 0.27   # normalized nose-to-shoulder distance; tune this

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


def analyze_head(nose_px, left_shoulder_px, right_shoulder_px, frame_height):
    """
    Returns whether the head is too close to the shoulders (forward head / computer neck).

    Logic: compute the vertical distance between the nose and the shoulder midpoint.
    When you slump forward, your head drops and this distance shrinks.
    Distance is normalized by frame height to be resolution-independent.
    """
    # Midpoint between the two shoulders
    mid_y = (left_shoulder_px[1] + right_shoulder_px[1]) / 2
    mid_x = (left_shoulder_px[0] + right_shoulder_px[0]) / 2

    # Vertical distance: shoulders are below nose in the frame (higher y value)
    # so mid_y > nose_y when posture is good; difference shrinks as head drops
    distance = (mid_y - nose_px[1]) / frame_height

    return {
        'bad_posture': distance < HEAD_DISTANCE_THRESHOLD,
        'distance': distance,
        'shoulder_mid_px': (int(mid_x), int(mid_y))  # useful for drawing
    }
