import cv2
import numpy as np
from src.motion_detection.CV_detecting import detect_motion_cv
from src.motion_detection.custom_detecting import detect_motion

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class App:
    '''This class gathers all the components together'''

    def __init__(self, video_src):
        self.prev_gray = None
        self.track_len = 25
        self.detect_interval = 10
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=180, varThreshold=120, detectShadows=False)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        if not self.cam.isOpened():
            raise ConnectionError("Не удалось подключиться к видеокамере")

        ret, prev_fr = self.cam.read()
        prev_fr = cv2.cvtColor(prev_fr, cv2.COLOR_BGR2GRAY)
        self.prev_gray = prev_fr[300:500, 500:700]

        while self.cam.isOpened():
            _ret, frame = self.cam.read()
            if not _ret:
                break
            cur_frame = frame[300:500, 500:700]
            cur_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cur_frame_gray = cur_frame_gray[300:500, 500:700]

            # frame_gray, vis, rect_centers, mask = detect_motion_cv(frame, bg_subtractor, kernel)
            frame_gray, vis, rect_centers, mask = detect_motion(cur_frame, cur_frame_gray, self.prev_gray)

            if self.prev_gray is None:
                self.prev_gray = frame_gray
                continue

            # Optical flow
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 5
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 0, 255))

            # Adding the rects centers as new points
            if self.frame_idx % self.detect_interval == 0:
                for center in rect_centers:
                    self.tracks.append([center])

            self.prev_gray = frame_gray
            cv2.imshow('Tracking', vis)
            cv2.imshow('Mask', mask)
            self.frame_idx += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            raise ValueError("Не удалось получить кадр")

        self.cam.release()
        cv2.destroyAllWindows()
