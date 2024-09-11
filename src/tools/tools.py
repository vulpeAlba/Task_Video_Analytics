from collections import deque
import numpy as np


def frame_difference(frame1, frame2, threshold=30):
    '''Returns the difference between two gray frames.
            diff_frame is a pixel array'''

    # Вычисляем разницу между двумя кадрами и применяем пороговое преобразование
    height = len(frame1)
    print(height)
    width = len(frame1[0])
    print(width)
    diff_frame = [[0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if frame2 is not None:
                diff = abs(frame2[y][x] - frame1[y][x])
                diff_frame[y][x] = 255 if diff > threshold else 0

    return diff_frame


def find_contours(diff_frame):
    '''Return the central coordinate of the rectangle.'''
    height = len(diff_frame)
    width = len(diff_frame[0])
    visited = [[False] * width for _ in range(height)]
    contours = []

    # Направления для смежных пикселей: вверх, вниз, влево, вправо, и по диагоналям
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def bfs(start_x, start_y):
        queue = deque([(start_x, start_y)])
        visited[start_y][start_x] = True
        contour = [(start_x, start_y)]

        while queue:
            x, y = queue.popleft()

            # Проверяем всех соседей
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx] and diff_frame[ny][nx] == 255:
                    visited[ny][nx] = True
                    queue.append((nx, ny))
                    contour.append((nx, ny))

        return contour

    # Поиск всех контуров
    for y in range(height):
        for x in range(width):
            if diff_frame[y][x] == 255 and not visited[y][x]:
                contour = bfs(x, y)
                contours.append(contour)

    return contours


def frame_to_pixel_array(frame):
    '''Converts the frame into a pixel array'''
    height, width, channels = frame.shape
    pixel_array = [[[frame[y, x, c] for c in range(channels)] for x in range(width)] for y in range(height)]
    return pixel_array


def pixel_array_to_frame(pixel_array):
    '''Converts the pixel array into a frame'''
    height = len(pixel_array)
    width = len(pixel_array[0])

    # Создаем пустой numpy массив для хранения изображения
    frame = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            frame[y, x] = pixel_array[y][x]

    return frame


def apply_threshold(frame, threshold):
    '''Filters the pixels of the frame by a specified values.
            Output frame is a pixel array'''
    height = len(frame)
    width = len(frame[0])
    frame = np.asarray(frame)

    for y in range(height):
        for x in range(width):
            gray_value = frame[y][x] # grayscale value
            if gray_value > threshold:
                frame[y][x] = 255
            else:
                frame[y][x] = 0
    return frame
