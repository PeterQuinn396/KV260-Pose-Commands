import numpy as np
import time
import cv2 as cv

import mediapipe as mp
import torch
from typing import List

from model_data.custom_dataset.model import get_model, CATEGORIES

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
        g = xir.Graph.deserialize("model_data/dhg_resnet/quantize_result/Gesture_Resnet.xmodel")
        subgraphs = get_child_subgraph_dpu(g)
        m = vart.Runner.createRunner(subgraphs[0], "run")

    else:
        print("Loading model...")
        m = get_model()
        m.load_state_dict(
            torch.load("model_data/custom_dataset/gesture_resnet18_custom_dataset.pt", map_location=device))
        m.eval()
        m.to(device)
        print('Done')

    return m


def run_inference(model, _input):
    input = _input.reshape(1, 3, 1080, 1920)

    if VITIS_DETECTED:

        # get pointers to IO data
        inputData = model.get_input_tensors()
        outputData = model.get_output_tensors()

        # load somehow?
        inputData[0] = input

        job_id = model.execute_async(inputData, outputData[0])
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


def main():
    model = load_model()
    alive = True

    cam = open_video()

    while cam.isOpened() and alive:
        start_time = time.time()

        frame = get_frame(cam)

        gesture = ""
        prob = ""
        if frame is not None:
            outputTensor = run_inference(model, frame)
            gesture, prob = process_output(outputTensor)

        end_time = time.time()
        dt = end_time - start_time
        alive = display_image(frame, gesture, prob, dt)

        send_message()


if __name__ == '__main__':
    main()
