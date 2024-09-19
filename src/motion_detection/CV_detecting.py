import random

import numpy as np

from src.tools.geometry import *
import cv2


def detect_motion_cv(frame, kernel, mask):
    '''Detect motion and exclude shadows using HSV color space'''
    mask_new = mask
    merged_rects = []
    vis = frame.copy()

    # Преобразуем кадр в HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Пороговые значения для исключения теней (настройте их при необходимости)
    lower_shadow = np.array([10, 10, 30])  # Низкий уровень насыщенности и яркости
    upper_shadow = np.array([170, 190, 255])  # Небольшая насыщенность и любая яркость

    # Создаём маску для исключения теней
    shadow_mask = cv2.inRange(hsv_frame, lower_shadow, upper_shadow)

    # Убираем тени из маски
    mask_new = cv2.bitwise_and(mask_new, cv2.bitwise_not(shadow_mask))

    # Морфологические операции для улучшения маски
    mask_new = cv2.morphologyEx(mask_new, cv2.MORPH_OPEN, kernel)
    mask_new = cv2.dilate(mask_new, None, iterations=2)  # Dilatation to fill the object

    _, thresh = cv2.threshold(mask_new, 150, 255, cv2.THRESH_BINARY) #120
    mask_new = cv2.morphologyEx(mask_new, cv2.MORPH_OPEN, kernel)  # Removing the noise
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Контуры
    bounding_rects = []
    rect_centers = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rects.append([x, y, x + w, y + h])

        # Объединяем близкие прямоугольники
        merged_rects = merge_rects(bounding_rects)

    # Находим центры прямоугольников
    for (x1, y1, x2, y2) in merged_rects:
        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        rect_centers.append(center)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY), vis, rect_centers, mask_new

