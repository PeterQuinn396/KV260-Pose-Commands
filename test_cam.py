import cv2 as cv



def open_video(cam_id=0):
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


    return cap


if __name__ == '__main__':
    cam = open_video(0)

    while True:
        success, image = cam.read()
        if not success:
            print("Got empty camera frame")

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv.flip(image, 1)

        cv.imshow("Image", image)

        if cv.waitKey(1) & 0xFF == 27:
            break

