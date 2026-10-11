import cv2 
import numpy as np

def detectColors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colors = {
        "red": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255])),
        ],
        "blue": [
            (np.array([100, 120, 70]), np.array([130, 255, 255])),
        ],
        "green": [
            (np.array([40, 70, 70]), np.array([85, 255, 255])),
        ],
    }
    mask_red = cv2.inRange(hsv, colors["red"][0][0], colors["red"][0][1])
    
    mask_blue = cv2.inRange(hsv, colors["blue"][0][0], colors["blue"][0][1])
    mask_green = cv2.inRange(hsv, colors["green"][0][0], colors["green"][0][1])
    masks = [(mask_red, (0, 0, 255)), (mask_blue, (255, 0, 0)), (mask_green, (0, 255, 0))]
    for mask, box_color in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    cv2.imshow("Red Mask", mask_red)
    cv2.imshow("Blue Mask", mask_blue)
    cv2.imshow("Green Mask", mask_green)


video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    if not ret:
        break
    detectColors(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
