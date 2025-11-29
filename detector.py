import cv2
import mediapipe as mp 
import numpy as np
import time
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

VELOCITY_THRESHOLD = 0.8    
SUSTAINED_FRAMES = 3
COOLDOWN = 2.0

KEYPOINTS = {
    'left_wrist': 15,
    'right_wrist': 16,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_shoulder': 11,
    'right_shoulder': 12
}

BUFFER_SIZE = 4

cap = cv2.VideoCapture(0)
time.sleep(1.0)

pos_buffers = {k: deque(maxlen=BUFFER_SIZE) for k in KEYPOINTS.keys()}
sustained_count = 0
last_alert = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        aggressive_flag = False

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            for name, idx in KEYPOINTS.items():
                lmx = lm[idx].x
                lmy = lm[idx].y
                pos_buffers[name].append((lmx, lmy))

 
            def velocity(buf):
                if len(buf) < 2:
                    return 0.0
                (x1,y1) = buf[-1]
                (x0,y0) = buf[-2]
                return np.sqrt((x1-x0)**2 + (y1-y0)**2)

            lw_vel = velocity(pos_buffers['left_wrist'])
            rw_vel = velocity(pos_buffers['right_wrist'])
            le_vel = velocity(pos_buffers['left_elbow'])
            re_vel = velocity(pos_buffers['right_elbow'])

            def forward_displacement(wrist_buf, shoulder_buf):
                if len(wrist_buf) < 2 or len(shoulder_buf) < 2:
                    return 0.0
                wx, wy = wrist_buf[-1]
                sx, sy = shoulder_buf[-1]
                return sx - wx  # positive if wrist is in front of shoulder (depends on camera orientation)

            left_forward = forward_displacement(pos_buffers['left_wrist'], pos_buffers['left_shoulder'])
            right_forward = forward_displacement(pos_buffers['right_wrist'], pos_buffers['right_shoulder'])

            # normalize forward threshold heuristically (values close to 0-1)
            # combine checks
            if (rw_vel > VELOCITY_THRESHOLD and right_forward > 0.02) or \
               (lw_vel > VELOCITY_THRESHOLD and left_forward > 0.02):
                aggressive_flag = True

        # sustained frames logic + cooldown for alerts
        if aggressive_flag:
            sustained_count += 1
        else:
            sustained_count = 0

        now = time.time()
        if sustained_count >= SUSTAINED_FRAMES and (now - last_alert) > COOLDOWN:
            last_alert = now
            print("[ALERT] Aggressive gesture detected at", time.strftime("%Y-%m-%d %H:%M:%S"))
            cv2.putText(frame, "ALERT: Aggressive gesture", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("Pose Aggression Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
