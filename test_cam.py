"""A small script to test webcam capture with opencv"""

import cv2


def open_video(cam_id=0):
    cap = cv2.VideoCapture(cam_id)
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


if __name__ == '__main__':
    cam = open_video(0)
    while True:
        success, image = cam.read()
        if not success:
            print("Got empty camera frame")

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.flip(image, 1)

        cv2.imshow("Image", image)

        print(image.shape)
        if cv2.waitKey(1) & 0xFF == 27:
            cam.release()
            break

