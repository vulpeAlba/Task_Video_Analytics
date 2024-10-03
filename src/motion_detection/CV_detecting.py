import numpy as np

from src.tools.geometry import *
import cv2


'''# Функция для вычисления HOG дескрипторов и их суммирования
def compute_hog_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return np.mean(hog_image)


def filter_shadows_hsv(frame, s_threshold=255, v_threshold=255, h_threshold=255):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Фильтрация по низкой насыщенности и яркости для теней
    shadow_mask = (s < s_threshold) & (v < v_threshold) & (h < h_threshold)
    return shadow_mask.astype(np.uint8) * 255
'''


def detect_motion_cv(frame, mask):
    '''Detect motion and filter shadows based on HOG and HSV'''

    vis = frame.copy()
    bounding_rects = []

    # Нахождение контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_centers = []
    for contour in contours:
        if cv2.contourArea(contour) < 600:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bounding_rects.append([x, y, x + w, y + h])

        merged_rects = merge_rects(bounding_rects)

        for (x1, y1, x2, y2) in merged_rects:
            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            rect_centers.append(center)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return vis, rect_centers, mask


def structure_tensor(frame, window_size=7):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Вычисляем градиенты по x и y
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Усреднение значений на основе окна
    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)

    # Вычисление следа и детерминанта матрицы
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy

    # Вычисление карты структуры (анизотропия текстуры)
    structure_map = det / (trace + 1e-6)  # Чтобы избежать деления на ноль

    return structure_map


def compute_structure_tensor(image, ksize=3, sigmaX=1):
    # Вычисляем градиенты по x и y
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Вычисляем компоненты тензора
    Ixx = grad_x * grad_x
    Ixy = grad_x * grad_y
    Iyy = grad_y * grad_y

    # Сглаживаем компоненты тензора
    Ixx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigmaX=sigmaX)
    Ixy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigmaX=sigmaX)
    Iyy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigmaX=sigmaX)

    return Ixx, Ixy, Iyy


def compute_eigenvalues(Ixx, Ixy, Iyy):
    # Собственные значения для каждого пикселя
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    sqrt_term = np.sqrt(np.maximum(trace ** 2 - 2 * det, 0))

    eigenvalue1 = (trace + sqrt_term) / 8.0
    eigenvalue2 = (trace - sqrt_term) / 8.0

    return eigenvalue1, eigenvalue2


def filter_shadows_hsv1(frame):
    # Преобразование изображения в HSV

    lower_shadow = np.array([0, 0, 0])  # Низкий уровень насыщенности и яркости
    upper_shadow = np.array([150, 80, 255])

    # Создаём маску для исключения теней
    shadow_mask = cv2.inRange(frame, lower_shadow, upper_shadow)

    return shadow_mask


def detect_motion_with_structure_tensor(frame, bg_subtractor, kernel, eigenvalue_threshold=120, ksize=3, sigmaX=2):
    '''Detect motion and filter shadows using structure tensor and HSV'''

    # Преобразуем изображение в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применяем фоновую разницу
    mask = bg_subtractor.apply(frame)

    # Морфологические операции для улучшения маски
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, None, iterations=3)

    # Вычисляем структурный тензор
    Ixx, Ixy, Iyy = compute_structure_tensor(gray, ksize=ksize, sigmaX=sigmaX)

    # Вычисляем собственные значения
    eigenvalue1, eigenvalue2 = compute_eigenvalues(Ixx, Ixy, Iyy)

    # Фильтруем области с низкими собственными значениями
    structure_mask = (eigenvalue1 > eigenvalue_threshold) & (eigenvalue2 > eigenvalue_threshold)
    structure_mask = structure_mask.astype(np.uint8) * 255

    # Фильтрация теней с помощью HSV
    shadow_mask = filter_shadows_hsv1(frame)

    # Инвертируем маску теней (где тень, там 0)
    shadow_mask = cv2.bitwise_not(shadow_mask)

    # Комбинируем маски движения и структуры
    combined_mask = cv2.bitwise_and(mask, structure_mask)
    combined_mask = cv2.bitwise_and(combined_mask, shadow_mask)
    combined_mask = cv2.dilate(combined_mask, None, iterations=3)

    return combined_mask



