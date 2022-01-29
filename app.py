import numpy as np
import time
import cv2 as cv
import mediapipe as mp
import torch

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

from model_data.dhg.model import HandGestureNet
from model_data.dhg.dataloader import CATEGORIES

from model_data.common import device

from hand_detector import HandDetector

VITIS_DETECTED = True

try:
    import xir
    import vart
except ModuleNotFoundError:
    print("No Vitis python packages found")
    print("Using CPU mode")
    VITIS_DETECTED = False


def load_model():
    print("Loading model...")
    m = HandGestureNet(n_channels=66, n_classes=14)
    m.load_state_dict(torch.load("model_data/dhg/gesture_pretrained_model_old_fmt.pt", map_location=device))
    m.eval()
    m.to(device)
    print('Done')

    # if VITIS ...

    return m


def run_inference(model, input):
    # if VITIS
    # ...

    return model(input.unsqueeze(0))


def process_output(outputTensor, detection_threshold=.5):
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


class HandLandmarkBuffer():
    def __init__(self, size: int = 100, channels: int = 66):
        self.buffer = torch.zeros((size, channels))

    def add(self, x:np.ndarray):
        # interpolate to produce an extra frame

        flat = x.flatten()
        next = torch.Tensor(flat)
        extra = .5 * (torch.add(next, self.buffer[-1]))
        self.buffer = torch.cat((self.buffer[2:], extra.unsqueeze(0), next.unsqueeze(0)), dim=0)


def main():

    model = load_model()
    alive = True

    hand_detector = HandDetector()
    cam = hand_detector.open_video(0)

    data_buffer = HandLandmarkBuffer()

    while cam.isOpened() and alive:
        start_time = time.time()

        image, im_cropped, world_space_points = hand_detector.get_cropped_hand()
        data_buffer.add(world_space_points)

        gesture = ""
        prob = ""
        if im_cropped is not None:
            outputTensor = run_inference(model, data_buffer.buffer)
            gesture, prob = process_output(outputTensor)

        end_time = time.time()
        dt = end_time - start_time
        alive = display_image(image, gesture, prob, dt)

        send_message()


if __name__ == '__main__':
    main()
