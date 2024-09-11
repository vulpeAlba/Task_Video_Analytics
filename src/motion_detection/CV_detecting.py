import random

from src.tools.geometry import *
import cv2


def detect_motion_cv(frame, bg_subtractor, kernel):
    '''Detect the motion by converting the frame to grayscale,
    finding the difference between two frames, applying the threshold
    and drawing contours around the moving object'''

    merged_rects = []
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()
    gray = frame_gray.copy()

    # Blurring the frames
    blured_frame = cv2.GaussianBlur(frame, ksize=(13, 13), sigmaX=0)

    # The font subtraction and threshold
    mask = bg_subtractor.apply(blured_frame)

    # Morphological operations to improve the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, None, iterations=3)  # Dilatation to fill the object

    _, thresh = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Removing the noise
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contours
    bounding_rects = []
    rect_centers = []

    for contour in contours:
        if cv2.contourArea(contour) < 1500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rects.append([x, y, x + w, y + h])

        # Merging the nearest rectangles
        merged_rects = merge_rects(bounding_rects)

    # Finding the centers
    for (x1, y1, x2, y2) in merged_rects:
        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        rect_centers.append(center)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return gray, vis, rect_centers, mask


