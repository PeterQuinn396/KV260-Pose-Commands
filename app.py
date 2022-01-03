import numpy as np
import time
import cv2 as cv
import mediapipe as mp
import torch
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# model specifics
from model.model import GestureClassifyModel
from model.asl_dataset import preprocess, CATEGORIES
from common import device

VITIS_DETECTED = True

try:
    import xir
    import vart
except ModuleNotFoundError:
    print("No Vitis python packages found")
    print("Using CPU mode")
    VITIS_DETECTED = False


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

        # make a square
        d = max(w, h)
        pad = d / 3
        x1 = int(max(0, x - pad))
        y1 = int(max(0, y - pad))
        x2 = int(min(im_width, x + d + pad))
        y2 = int(min(im_height, y + d + pad))

        cropped_im = image[y1:y2, x1:x2].copy()  # slice bounding box of hand
        # draw the rectange on the image in place
        cv.rectangle(im_cp, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return im_cp, cropped_im


def load_model():
    print("Loading model...")
    model = GestureClassifyModel("model/asl_resnext.pth")
    model.to(device)
    print("Done")
    return model


def run_inference(model, image):
    # convert to PIL image
    with torch.no_grad():
        cv.imshow("crop", cv.cvtColor(image, cv.COLOR_RGB2BGR))
        cv.waitKey(1)

        input = Image.fromarray(image, 'RGB')
        input = preprocess(input)
        input = input.unsqueeze(0)  # model expects batch i.e. [BATCH, C, H, W]
        input.to(device)
        return model.forward(input)


def process_output(outputTensor, detection_threshold=.2):
    out = outputTensor.squeeze()
    probs = torch.softmax(out, dim=0)
    max = torch.max(probs)
    if max > detection_threshold:
        gesture = CATEGORIES[torch.argmax(probs)]
    else:
        gesture = None
    return gesture, max


def send_message():
    pass


def display_image(image, gesture, probability, time) -> bool:
    # input is rgb
    im = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    # im = image.copy()

    if time > 0:  # avoid a crash from a bad time
        cv.putText(im, f"FPS: {1 / time:.2f}", (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv.putText(im, f"Gesture: {gesture}, Prob: {probability}", (5, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

    cv.imshow("Image", im)
    alive = True
    if cv.waitKey(1) & 0xFF == 27:
        alive = False
    return alive


def main():
    cam = open_video(cam_id=0)

    model = load_model()
    alive = True
    hands = mp_hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5, max_num_hands=1)

    while cam.isOpened() and alive:
        start_time = time.time()
        image = get_frame(cam)

        image, im_cropped = crop_hand(image, hands)

        gesture = ""
        prob = ""
        if im_cropped is not None:
            outputTensor = run_inference(model, im_cropped)
            gesture, prob = process_output(outputTensor)

        end_time = time.time()
        dt = end_time - start_time
        alive = display_image(image, gesture, prob, dt)

        send_message()


if __name__ == '__main__':
    main()
