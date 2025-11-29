import cv2
import numpy as np
import imutils 
import time

MIN_CONTOUR_AREA = 2000   
MOVEMENT_FRAMES_REQUIRED = 3  
ALERT_COOLDOWN = 2.0     

cap = cv2.VideoCapture(0) 
time.sleep(1.0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

movement_count = 0
last_alert_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_area = False
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            big_area = True

    if big_area:
        movement_count += 1
    else:
        movement_count = 0

    # if sustained movement → flag aggressive movement
    if movement_count >= MOVEMENT_FRAMES_REQUIRED:
        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN:
            last_alert_time = now
            cv2.putText(frame, "ALERT: Aggressive movement!", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            print("[ALERT]", time.strftime("%Y-%m-%d %H:%M:%S"))

    cv2.imshow("frame", frame)
    cv2.imshow("mask", fgmask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

