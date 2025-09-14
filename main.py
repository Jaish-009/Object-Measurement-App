import cv2
import numpy as np

# --------------- Settings ----------------
# Known width of reference object in cm (e.g., Credit card = 8.5 cm)
REF_OBJECT_WIDTH = 8.5  

cap = cv2.VideoCapture(0)

# Helper function
def distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Calibration step
calibrated = False
pixels_per_cm = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:  # ignore small noise
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        # Extract dimensions in pixels
        (tl, tr, br, bl) = box
        dA = distance(tl, bl)   # height in px
        dB = distance(tr, br)   # width in px

        if not calibrated:
            # Assume first detected object is the reference
            pixels_per_cm = dB / REF_OBJECT_WIDTH
            calibrated = True
            print(f"Calibrated: {pixels_per_cm:.2f} pixels/cm")

        if calibrated and pixels_per_cm is not None:
            obj_height = dA / pixels_per_cm
            obj_width = dB / pixels_per_cm

            cv2.putText(frame, "{:.1f} cm".format(obj_width),
                        (int(tr[0]), int(tr[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "{:.1f} cm".format(obj_height),
                        (int(tl[0]), int(tl[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Object Measurement (No ArUco)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
