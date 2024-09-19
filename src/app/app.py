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
        self.track_len = 25  # 15
        self.detect_interval = 10  # 20
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=110, varThreshold=130)  # 180, 120
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        ret, prev_fr = self.cam.read()
        prev_mask = bg_subtractor.apply(prev_fr)
        prev_fr = prev_fr[300:500, 500:700]
        prev_gray = cv2.cvtColor(prev_fr, cv2.COLOR_BGR2GRAY)

        iteri = 0

        if not self.cam.isOpened():
            raise ConnectionError("Не удалось подключиться к видеокамере")

        while self.cam.isOpened():
            _ret, frame = self.cam.read()
            if not _ret:
                break
            cur_frame = frame[300:500, 500:700]
            cur_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cur_frame_gray = cur_frame_gray[300:500, 500:700]

            if iteri == 0 or iteri % 5 == 0:
                mask = bg_subtractor.apply(frame)
            else:
                mask = prev_mask

            frame_gray, vis, rect_centers, mask_new = detect_motion_cv(frame, kernel, mask)
            # frame_gray, vis, rect_centers, mask_new = detect_motion(cur_frame, prev_fr)

            # Optical flow
            vis = self.run_optical_flow(prev_gray, frame_gray, vis, rect_centers)

            prev_gray = frame_gray
            prev_fr = cur_frame
            cv2.imshow('Tracking', vis)
            cv2.imshow('Mask', mask_new)
            self.frame_idx += 1
            iteri += 1
            prev_mask = mask

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            raise ValueError("Не удалось получить кадр")

        self.cam.release()
        cv2.destroyAllWindows()


    def run_optical_flow(self, prev_gray, frame_gray, vis, rect_centers):
        if len(self.tracks) > 0:
            img0, img1 = prev_gray, frame_gray
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

        return vis

