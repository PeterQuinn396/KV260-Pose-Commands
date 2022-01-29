import cv2 as cv
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class HandDetector():

    def __init__(self):
        self.cam = 0
        self.hands = mp_hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5, max_num_hands=2)

    def open_video(self, cam_id=0):
        cap = cv.VideoCapture(cam_id)
        if not cap.isOpened():
            print("Error: cannot open camera")
            exit(-1)

        # grab test frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            exit(-1)

        self.cam = cap

        return cap

    def _get_frame(self):
        success, image = self.cam.read()
        if not success:
            print("Got empty camera frame")

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        return image

    def _crop_hand(self, image):
        im_height, im_width, _ = image.shape

        # inference (might) run faster if image is read only
        image.flags.writeable = False
        results = self.hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        im_cp = image.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(im_cp, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cropped_im = None
        if results.multi_hand_landmarks:
            points = results.multi_hand_landmarks[0].landmark
            cnt = [[pt.x * im_width, pt.y * im_height] for pt in points]
            cnt = np.array(cnt, int)
            x, y, w, h = cv.boundingRect(cnt)

            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # make a square
            d = max(w, h)
            pad = 60
            x1 = int(max(0, center_x - d / 2 - pad))
            y1 = int(max(0, center_y - d / 2 - pad))
            x2 = int(min(im_width, center_x + d / 2 + pad))
            y2 = int(min(im_height, center_y + d / 2 + pad))

            cropped_im = image[y1:y2, x1:x2].copy()  # slice bounding box of hand
            # draw the rectange on the image in place
            cv.rectangle(im_cp, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return im_cp, cropped_im, results

    def augment_world_space_data(self, hand_landmark_data):
        world_space_coords = hand_landmark_data.multi_hand_world_landmarks[0].landmark
        arr = [[pt.x, pt.y, pt.z] for pt in world_space_coords]
        arr = np.array(arr)
        index_arr = np.array([0, 9])
        palm_center = np.mean(arr[index_arr])
        arr = np.insert(arr, 1, palm_center, axis=0)

        return arr

    def get_cropped_hand(self):
        im = self._get_frame()
        im_cp, cropped_im, hand_landmark_data = self._crop_hand(im)

        if not hand_landmark_data.multi_hand_world_landmarks:
            world_space_coords = np.zeros((22,3))
        else:
            world_space_coords = self.augment_world_space_data(hand_landmark_data)

        return im_cp, cropped_im, world_space_coords
