import numpy as np
import time
import cv2 as cv

import mediapipe as mp
import torch
from typing import List

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

from model_data.dhg_resnet.model import load_model as get_model
from model_data.dhg_resnet.dataloader import CATEGORIES

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


# not sure what this for
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def load_model():

    if VITIS_DETECTED:
        g = xir.Graph.deserialize("modle_data/dhg_resnet/quantize_result/Gesture_Resnet.xmodel")
        subgraphs = get_child_subgraph_dpu(g)
        m = vart.Runner.createRunner(subgraphs[0], "run")

    else:
        print("Loading model...")
        m = get_model()
        m.load_state_dict(torch.load("model_data/dhg_resnet/gesture_resnet18_model_new_fmt.pt", map_location=device))
        m.eval()
        m.to(device)
        print('Done')

    return m


def run_inference(model, _input):

    input = _input.reshape(1,3,100,22)

    if VITIS_DETECTED:

        # get pointers to IO data
        inputData = model.get_input_tensors()
        outputData = model.get_output_tensors()

        # load somehow?
        inputData[0] = input

        job_id = model.execute_async(inputData,outputData[0])
        model.wait(job_id)
        y = outputData

    else:
        y = model(input)

    return y


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
