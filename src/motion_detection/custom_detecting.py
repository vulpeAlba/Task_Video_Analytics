import cv2

from src.tools.geometry import get_center, merge_rects
from src.tools.image import *
from src.tools.tools import *


def detect_motion(frame, prev_fr):
    '''Detect the motion by converting the frame to grayscale,
    finding the difference between two frames, applying the threshold
    and drawing contours around the moving object'''

    # Converting the frames to grayscale
    roi = frame.copy()
    roi_current = rgb_to_grayscale(frame)

    rect_centers = []

    roi_prev = rgb_to_grayscale(prev_fr)
    # Subtraction of the frames

    roi_prev_1 = box_filter(pixel_array_to_frame(roi_prev), 3)
    roi_current_1 = box_filter(pixel_array_to_frame(roi_current), 3)

    diff_frame = frame_difference(roi_current_1, roi_prev_1)
    mask = pixel_array_to_frame(diff_frame)
    # Threshold
    tresh_mask = apply_threshold(pixel_array_to_frame(diff_frame), 250)
    # Contours drawing
    contours = find_contours(tresh_mask)
    rects = get_bounding_rectangles(contours)
    merged_rects = merge_rects(rects)

    for cnt in merged_rects:
        rect_centers.append(get_center(cnt))

    roi = draw_rectangles(roi, contours)
    cv2.imshow("roi", roi)

    return pixel_array_to_frame(roi_current), roi, rect_centers, mask
