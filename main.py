import cv2  # Open Source Computer Vision Library (uv pip install opencv-python)
import mediapipe as mp  # MediaPipe model
from posture_analyzer import analyze_shoulders
from alerter import posture_alerter

# set up alerter for bad posture
alerter = posture_alerter()

# setup MediaPipe Pose (loads model once and create a pose object to call on every frame)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,  # model certanty before reporting a detection (0.5 - sensible default)
                    min_tracking_confidence=0.5)  # model certanty in continuing to track the detection (0.5 - sensible default)

# Landmark indices (named indices into MediaPipe's 33-keypoint body map)
NOSE = mp_pose.PoseLandmark.NOSE
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER

# open camera
camera = cv2.VideoCapture(0)  # 0 means default camera (try 1,2,etc)
# check
if not camera.isOpened():
    print('Error: could not open the camera')
    exit()
print('Camera opened')

# loop continuously to read frames
while True:  # runs forever until q
    # get frame
    success, frame = camera.read()  # grab 1 frame (px grid img) + T/F
    # check
    if not success:
        print('Error: could not read frame')
        break
    
    # lauch MediaPipe
    h, w = frame.shape[:2]  # frame height and width in pixels
    # convert BGR -> RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # run pose detection
    results = pose.process(rgb_frame)

    # extract landmarks and their coords
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # get pixel coords for each point
        def get_px(landmark_idx):
            lm = landmarks[landmark_idx]
            return int(lm.x * w), int(lm.y * h)  # (x, y) in pixels
        nose_px = get_px(NOSE)
        left_sh_px = get_px(LEFT_SHOULDER)
        right_sh_px = get_px(RIGHT_SHOULDER)
        
        # posture analysis based on landmarks coords
        result = analyze_shoulders(left_sh_px, right_sh_px, h)

        # update alerter
        alerter.update(result['bad_posture'])

        # draw landmarks and posture analysis results
        color = (0, 0, 255) if result['bad_posture'] else (0, 255, 0)  # red = bad, green = good
        # draw circles on the three points
        cv2.circle(frame, nose_px, 8, (0, 255, 255), -1)  # yellow = nose
        cv2.circle(frame, left_sh_px,  8, color, -1)
        cv2.circle(frame, right_sh_px, 8, color, -1)
        cv2.line(frame, left_sh_px, right_sh_px, color, 2)

        # show tilt value and warning on screen
        tilt_text = f"Tilt: {result['tilt']:.3f}"
        cv2.putText(frame, tilt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if result['bad_posture']:
            cv2.putText(frame, "FIX YOUR SHOULDERS", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # display the frame in a window
    cv2.imshow('Posture Monitor', frame)  # prepare title + img to display (new frame each itteration)
    # show window (wait 1 ms for a key press. if user presses q, break loop and close)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing camera...")
        break

# release camera and close all windows
camera.release()  # so other programs can use it
pose.close()
cv2.destroyAllWindows()
print('Done')





