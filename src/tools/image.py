import random
import numpy as np
from src.tools.geometry import get_bounding_rectangles


def box_filter(frame, k):
    '''Applies a box filter and returns the blurred the image.'''

    height = len(frame)
    width = len(frame[0])
    frame = np.asarray(frame)

    # Размеры выходного изображения
    new_height = height - k + 1
    new_width = width - k + 1

    filtered_img = np.zeros((new_height, new_width))
    for i in range(new_height):
        for j in range(new_width):
            filtered_img[i, j] = np.mean(frame[i:i + k, j:j + k])

    return filtered_img


def rgb_to_grayscale(frame):
    '''Converts the image from color to black and white.'''

    height = len(frame)
    width = len(frame[0])

    grayscale_image = [[0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r, g, b = frame[y][x]
            grayscale_image[y][x] = int(0.299 * r + 0.587 * g + 0.114 * b)

    return grayscale_image


def draw_rectangles(frame, contours):
    '''Gets the contours' coordinates and draw rectangles on the frame.
    Returns the frame with rects.'''

    if not contours:
        return frame

    rects = get_bounding_rectangles(contours)
    for rect in rects:
        min_x, min_y, max_x, max_y = rect
        for x in range(min_x, max_x + 1):
            frame[min_y][x] = (0, 255, 0)  # Верхняя граница
            frame[max_y][x] = (0, 255, 0)  # Нижняя граница

        for y in range(min_y, max_y + 1):
            frame[y][min_x] = (0, 255, 0)  # Левая граница
            frame[y][max_x] = (0, 255, 0)  # Правая граница

    return frame


def get_random_color():
    '''Gets the random color (x1, x2, x3).'''
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
