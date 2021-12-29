import numpy as np
import time
import cv2 as cv
import mediapipe as mp
import torch

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# model specifics
from .model.model import GestureClassifyModel
from .model.asl_dataset import preprocess, CATEGORIES

VITIS_DETECTED = True
try:
    import xir
    import vart
except ModuleNotFoundError:
    print("No Vitis python packages found")
    print("Using CPU mode")
    VITIS_DETECTED = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # enable gpu processing


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


def crop_hand(image, hands):
    im_height, im_width, _ = image.shape

    # inference (might) run faster if image is read only
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    cropped_im = None
    if results.multi_hand_landmarks:
        points = results.multi_hand_landmarks[0].landmark
        cnt = [[pt.x * im_width, pt.y * im_height] for pt in points]
        cnt = np.array(cnt, int)
        x, y, w, h = cv.boundingRect(cnt)
        x1 = int(max(0, x - w / 10))
        y1 = int(max(0, y - h / 10))
        x2 = int(min(im_width, x + w + w / 10))
        y2 = int(min(im_height, y + h + h / 10))

        cropped_im = image[x1:x2, y1:y2]  # slice bounding box of hand
        # draw the rectange on the image in place
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return image, cropped_im


def load_model():
    model = GestureClassifyModel("model/asl_resnext.pth")
    return model


def run_inference(model, image):
    input = preprocess(image)
    return model.forward(input)


def process_output(outputTensor):
    gesture = CATEGORIES[torch.argmax(outputTensor)]
    return gesture


def send_message():
    pass


def display_image(image, gesture, t) -> bool:
    # input is rgb
    im = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    # im = image.copy()

    if t > 0:  # avoid a crash from a bad time
        cv.putText(im, f"FPS: {1 / t:.2f}", (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv.putText(im, f"Gesture: {gesture}", (5, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

    cv.imshow("Image", im)
    alive = True
    if cv.waitKey(1) & 0xFF == 27:
        alive = False
    return alive


def main():
    cam = open_video()

    model = load_model()
    alive = True
    hands = mp_hands.Hands(min_detection_confidence=.5,
                           min_tracking_confidence=.5,
                           max_num_hands=1)

    while cam.isOpened() and alive:
        start_time = time.time()
        image = get_frame(cam)

        image, im_cropped = crop_hand(image, hands)

        outputTensor = run_inference(model, im_cropped)

        gesture = process_output(outputTensor)

        end_time = time.time()
        dt = end_time - start_time
        alive = display_image(image, gesture, dt)

        send_message()


if __name__ == '__main__':
    main()
