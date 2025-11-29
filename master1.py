import cv2
import time
import imutils
import os
import datetime
from collections import deque

MIN_AREA = 1500              # min contour area to count as "big"
MOTION_FRAMES_REQUIRED = 3   # consecutive frames of big motion to trigger
ALERT_DISPLAY_SECONDS = 2.0  # how long alert stays visible
FLASH_INTERVAL = 0.25        # flash toggle interval (s)
PRE_SEC = 3                  # seconds to include before trigger in clip
POST_SEC = 3                 # seconds to include after trigger in clip
OUTPUT_DIR = "output_alerts" # where snapshots/clips are saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
fgbg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=25, detectShadows=False)

cap = cv2.VideoCapture(0)
time.sleep(0.5)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot open camera (cap.read() failed)")

# Try to get camera FPS, fallback to 20
cam_fps = cap.get(cv2.CAP_PROP_FPS)
if cam_fps is None or cam_fps <= 0 or cam_fps != cam_fps:  # nan check
    cam_fps = 20.0
FPS = float(cam_fps)
print(f"[INFO] Camera FPS detected: {FPS:.2f}")

# Standardize frame size (resize for speed)
FRAME_W = 720
frame = imutils.resize(frame, width=FRAME_W)
H, W = frame.shape[:2]

PRE_FRAMES = deque(maxlen=int(PRE_SEC * FPS))

motion_sustain = 0
alert_active = False
last_alert_time = 0.0

flash_on = False
last_flash_time = 0.0

recording_writer = None
post_frames_left = 0
current_clip_name = None

def timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_snapshot(img_bgr):
    fname = os.path.join(OUTPUT_DIR, f"snapshot_{timestamp_str()}.jpg")
    cv2.imwrite(fname, img_bgr)
    print("[SAVED] snapshot ->", fname)
    return fname

def start_clip_writer(pre_frames, first_frame):
    fname = os.path.join(OUTPUT_DIR, f"clip_{timestamp_str()}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(fname, fourcc, FPS, (W, H))
    # write pre buffer
    for f in pre_frames:
        # ensure size
        writer.write(imutils.resize(f, width=FRAME_W))
    # write the triggering frame too
    writer.write(imutils.resize(first_frame, width=FRAME_W))
    print("[SAVED] clip started ->", fname)
    return writer, fname

try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] empty frame, skipping")
            time.sleep(0.01)
            continue

        frame = imutils.resize(frame, width=FRAME_W)
        frame = cv2.flip(frame, 1)  # mirror for natural preview
        original = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Auto-blur faces (privacy)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(face, (35, 35), 20)

        # 2) Motion detection
        fgmask = fgbg.apply(gray)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_rects = []
        for c in contours:
            if cv2.contourArea(c) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(c)
                big_rects.append((x, y, w, h))

        # 3) Pre-buffer original frames (unblurred for recording clarity)
        PRE_FRAMES.append(original.copy())

        # 4) Sustained motion logic
        if big_rects:
            motion_sustain += 1
        else:
            motion_sustain = 0

        aggressive_motion = motion_sustain >= MOTION_FRAMES_REQUIRED
        now = time.time()

        # If aggressive and not already in alert: trigger snapshot + start clip writer
        if aggressive_motion and not alert_active:
            alert_active = True
            last_alert_time = now

            # snapshot (save original, unblurred to retain context — change if you prefer blurred)
            save_snapshot(original)

            # start clip writer and write pre-buffer
            recording_writer, current_clip_name = start_clip_writer(list(PRE_FRAMES), original)
            post_frames_left = int(POST_SEC * FPS)

        # If currently recording post frames, continue writing and decrement counter
        if recording_writer is not None:
            try:
                recording_writer.write(original)
            except Exception as e:
                print("[ERROR] writing to clip:", e)
            post_frames_left -= 1
            if post_frames_left <= 0:
                try:
                    recording_writer.release()
                except:
                    pass
                print("[INFO] finished clip:", current_clip_name)
                recording_writer = None
                current_clip_name = None

        # Manage alert duration and flashing
        if alert_active and (now - last_alert_time > ALERT_DISPLAY_SECONDS):
            alert_active = False

        if now - last_flash_time > FLASH_INTERVAL:
            flash_on = not flash_on
            last_flash_time = now

        # Draw bounding boxes: flashing if alert_active else simple
        display = frame.copy()
        for (x, y, w, h) in big_rects:
            if alert_active and flash_on:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0,0,255), 4)
            elif alert_active:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0,0,150), 2)
            else:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,255), 1)

        # Overlay status text + last saved names
        cv2.putText(display, f"Status: {'ALERT' if alert_active else 'OK'}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255) if alert_active else (0,200,0), 2)

        # Show last saved filenames on-screen (if any)
        # (list files in OUTPUT_DIR and show most recent if present)
        try:
            files = sorted(os.listdir(OUTPUT_DIR))
            if files:
                recent = files[-1]
                cv2.putText(display, f"Last saved: {recent}", (10, H-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        except Exception:
            pass

        # Display
        cv2.imshow("Aggressive Motion Detector", display)

        # Frame rate control (approx)
        elapsed = time.time() - start_time
        delay = max(0, (1.0 / FPS) - elapsed)
        time.sleep(delay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
finally:
    # cleanup
    if recording_writer is not None:
        try:
            recording_writer.release()
        except:
            pass
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exiting, resources released.")



