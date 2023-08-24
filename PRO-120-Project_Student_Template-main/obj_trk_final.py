import cv2
import sys
import numpy as np

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2, 1)
    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Record the center of the bounding box as a trajectory point
    center_x = x + w // 2
    center_y = y + h // 2
    trajectory.append((center_x, center_y))

cap = cv2.VideoCapture('footvolleyball.mp4')  # Replace with actual video path
trajectory = []  # Initialize an empty trajectory list

# Tracker initialization
tracker = cv2.TrackerCSRT_create()
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
tracker.init(img, bbox)

while True:
    check, img = cap.read()

    success, bbox = tracker.update(img)

    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "LOST", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw the trajectory on the frame
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            cv2.line(img, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

    # Display the trajectory on a separate image
    if len(trajectory) > 1:
        traj_image = np.zeros_like(img)
        for i in range(1, len(trajectory)):
            cv2.line(traj_image, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)
        cv2.imshow("Trajectory", traj_image)

    cv2.imshow("Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 