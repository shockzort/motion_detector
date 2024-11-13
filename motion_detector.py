import cv2
import numpy as np

test_video_path = "/home/shockzor/workbench/test_videos/1.mp4"


class motion_detector:
    def __init__(
        self,
        min_detecteble_motion_area: int = 32 * 32,
        thresholding_diff: int = 20,
        gaussian_blur_size: tuple = (5, 5),
        gaussian_sigma: int = 0,
        dilate_kernel_size: np.array = np.ones((5, 5)),
        dilate_iterations: int = 1,
        alpha: float = 0.1
    ):
        self.min_detecteble_motion_area = min_detecteble_motion_area
        self.thresholding_diff = thresholding_diff
        self.gaussian_blur_size = gaussian_blur_size
        self.gaussian_sigma = gaussian_sigma
        self.dilate_kernel_size = dilate_kernel_size
        self.dilate_iterations = dilate_iterations
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.avg_image = None
        self.prev_frame = None

    def _preprocess_frame(self, frame):
        preprocessed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(
            preprocessed, ksize=self.gaussian_blur_size, sigmaX=self.gaussian_sigma
        )

    def _update_average(self, image):
        if self.avg_image is None:
            self.avg_image = image
        else:
            self.avg_image = cv2.addWeighted(image, self.alpha, self.avg_image, self.beta, 0.0)
        return self.avg_image

    def _process_frame(self, prev, cur):
        difference_frame = cv2.absdiff(src1=prev, src2=cur)
        difference_frame = cv2.dilate(
            difference_frame, self.dilate_kernel_size, iterations=self.dilate_iterations
        )
        thresholded_frame = cv2.threshold(
            src=difference_frame,
            thresh=self.thresholding_diff,
            maxval=255,
            type=cv2.THRESH_BINARY,
        )[1]

        contours, _ = cv2.findContours(
            thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        sum_contours_area = 0
        detected_motion_rects = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Ignore small contours
            if area > self.min_detecteble_motion_area:
                sum_contours_area += area

                x, y, w, h = cv2.boundingRect(contour)
                detected_motion_rects.append([x, y, w, h])

        return sum_contours_area >= self.min_detecteble_motion_area, detected_motion_rects

    def detect_motion(self, image):
        if image is None:
            print("Cannot process image, image is None")
            return False, []

        cur_preprocessed = self._preprocess_frame(image)

        if self.prev_frame is None:
            self.prev_frame = cur_preprocessed
            return False, []

        motion, rects = self._process_frame(self.prev_frame, cur_preprocessed)
        self.prev_frame = cur_preprocessed
        return motion, rects

def process_video(md: motion_detector, video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video {}".format(video_path))
        return

    while cap.isOpened():
        ok, frame = cap.read()
        if ok:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            has_motion, rects = md.detect_motion(frame)

            if has_motion:
                for rect in rects:
                    cv2.rectangle(
                        frame,
                        (rect[0], rect[1]),
                        (rect[0] + rect[2], rect[1] + rect[3]),
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

            cv2.imshow(video_path, frame)
            print("processed frame {}".format(int(pos_frame)))

        if cv2.waitKey(1) == 27:
            break


md = motion_detector()
process_video(md, test_video_path)