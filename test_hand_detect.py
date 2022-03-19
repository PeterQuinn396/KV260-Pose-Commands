"""A small script to open the webcam and test the mediapipe hand detection."""

import cv2 as cv
import numpy as np
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def open_video():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera")
        exit(-1)

    # grab test frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit(-1)

    return cap


def process_video(cam: cv.VideoCapture):
    frame_rate_buffer_size = 10
    frame_rates = np.zeros(frame_rate_buffer_size)
    ind = 0
    with mp_hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5) as hands:
        while cam.isOpened():

            start_time = time.time()

            success, image = cam.read()
            if not success:
                print("Ignoring empty camera frame")
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)

            # calculate frame rate
            frame_rates[ind] = time.time() - start_time
            ind += 1
            ind = ind % frame_rate_buffer_size
            fps = np.average(frame_rates)
            fps = 1 / (fps)
            cv.putText(image, 'FPS: ' + str(int(fps)), (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

            cv.imshow('MediaPipe Hands', image)
            if cv.waitKey(1) & 0xFF == 27:
                cam.release()
                cv.destroyAllWindows()

    exit()


if __name__ == "__main__":
    cam = open_video()
    process_video(cam)
