
'''min_contour_area = 200
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    rect_centers = []
    rects = []

    for i, cnt1 in enumerate(large_contours):
        rect1 = cv2.boundingRect(cnt1)
        for j, cnt2 in enumerate(large_contours[i + 1:], i + 1):
            rect2 = cv2.boundingRect(cnt2)
            rects.append(rect1)
            rects.append(rect2)

            if intersect(rect1, rect2):
                merged_rectangle = merge_rectangles(rect1, rect2)
                rects.append(merged_rectangle)
                rect_centers.append(get_center(merged_rectangle))
            if contains(rect1, rect2):
                merged_rectangle = merge_rectangles(rect1, rect2)
                rects.append(merged_rectangle)
                rect_centers.append(get_center(merged_rectangle))

    # for i in range(len(rects) - 1):
       # if intersect(rects[i], rects[i+1]):
           # merged_rectangle = merge_rectangles(rects[i], rects[i+1])
           # rects.append(merged_rectangle)
           # rect_centers.append(get_center(merged_rectangle))

    # Drawing merged rectangles
    frame_out = frame.copy()
    bounding_rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.2)

    for (x1, y1, x2, y2) in bounding_rects:
        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        rect_centers.append(center)
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Output frames
    cv2.imshow("Frame", frame_out)
    cv2.imshow("Mask", mask)

    return frame_out, rect_centers'''