from math import sqrt


def create_bounding_rect(contour):
    '''Returns the coordinates(pixels) for bounding rectangles drawing'''
    x_min = min([p[0] for p in contour])
    y_min = min([p[1] for p in contour])
    x_max = max([p[0] for p in contour])
    y_max = max([p[1] for p in contour])
    return (x_min, y_min, x_max, y_max)


def get_bounding_rectangles(contours):
    '''Returns the array of each rect's coordinates'''

    rectangles = []
    for contour in contours:
        rect = create_bounding_rect(contour)
        rectangles.append(rect)
    return rectangles


def intersect(rect1, rect2) -> bool:
    '''Checks if the rectangles intersect.
     Rectangles are [x1, y1, x2, y2]'''

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def contains(big_rect, small_rect) -> bool:
    '''Checks if `big_rect` contains `small_rect`.
    Rectangles are [x1, y1, x2, y2]'''

    x1, y1, w1, h1 = big_rect
    x2, y2, w2, h2 = small_rect
    return x1 <= x2 and w2 <= w1 and y1 <= y2 and h2 <= h1


def find_near_rects(rect1_center, rect2_center) -> bool:
    '''Checks if centers of the rectangles are nearby.
            Rectangles are [x1, y1, x2, y2]'''

    x1, y1 = rect1_center
    x2, y2 = rect2_center
    distance = 20
    return sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2)) <= distance


def merge_rects(rects):
    '''Merges the rectangles into the biggest possible one.
                Rectangles are [x1, y1, x2, y2]'''
    def rects_overlap(r1, r2):
        return not (r1[0] > r2[2] or r1[2] < r2[0] or r1[1] > r2[3] or r1[3] < r2[1])

    merged = []
    for r in rects:
        found_overlap = False
        for i, m in enumerate(merged):
            if rects_overlap(m, r):
                merged[i] = [min(m[0], r[0]), min(m[1], r[1]), max(m[2], r[2]), max(m[3], r[3])]
                found_overlap = True
                break
        if not found_overlap:
            merged.append(r)
    return merged


def get_center(rect):
    '''Returns the central coordinate of the rectangle.
            Rectangles are [x1, y1, x2, y2]'''

    return [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2]
