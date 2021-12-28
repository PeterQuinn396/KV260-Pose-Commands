import numpy as np
import time
import cv2 as cv

VITIS_DETECTED = True
try:
    import xir
    import vart
except ModuleNotFoundError:
    print("No Vitis python packages found")
    print("Using CPU mode")
    VITIS_DETECTED = False


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


def get_frame(cam):
    success, image = cam.read()
    if not success:
        print("Got empty camera frame")

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    return image


def load_model():
    pass


def run_inference(model, image):
    pass


def process_output(outputTensor):
    gesture = ""
    return gesture


def send_message():
    pass


def display_image(image, gesture, t) -> bool:
    # input is rgb
    im = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    #im = image.copy()


    if t > 0:
        cv.putText(im, f"FPS: {1 / t:.2f}", (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv.putText(im, f"Gesture: ", (5, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

    cv.imshow("Image", im)
    alive = True
    if cv.waitKey(1) & 0xFF == 27:
        alive = False
    return alive


def main():
    cam = open_video()

    model = load_model()
    alive = True
    while cam.isOpened() and alive:
        start_time = time.time()
        image = get_frame(cam)

        # inference (might) run faster if image is read only
        image.flags.writeable = False
        outputTensor = run_inference(model, image)
        gesture = process_output(outputTensor)

        # Draw the hand annotations on the image.
        image.flags.writeable = True

        end_time = time.time()
        dt = end_time - start_time
        alive = display_image(image, gesture, dt)

        send_message()


if __name__ == '__main__':
    main()
