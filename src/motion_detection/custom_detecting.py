import cv2

from src.tools.geometry import get_center
from src.tools.image import *
from src.tools.tools import *


def detect_motion(frame, frame_gray, prev_fr):
    '''Detect the motion by converting the frame to grayscale,
    finding the difference between two frames, applying the threshold
    and drawing contours around the moving object'''

    # Converting the frames to grayscale
    roi_current = frame_gray
    roi = frame
    roi_prev = None

    rect_centers = []

    if prev_fr is not None:
        roi_prev = prev_fr
        # Subtraction of the frames

    diff_frame = frame_difference(roi_current, roi_prev)
    mask = pixel_array_to_frame(diff_frame)
    # Threshold
    tresh_mask = apply_threshold(pixel_array_to_frame(diff_frame), 254)
    # Contours drawing
    contours = find_contours(tresh_mask)
    rects = get_bounding_rectangles(contours)
    for cnt in rects:
        rect_centers.append(get_center(cnt))

    roi = draw_rectangles(roi, contours)
    cv2.imshow("roi", roi)

    return roi_current, roi, rect_centers, mask
