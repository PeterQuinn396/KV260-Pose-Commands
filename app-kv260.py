import os.path

import numpy as np
import time
import cv2 as cv
import torch
from typing import List

from model_data.custom_dataset.model import CATEGORIES

from pynq_dpu import DpuOverlay

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
    
    overlay = DpuOverlay("dpu.bit")
    path = "model_data/custom_dataset/Resnet_custom_kv260.xmodel"
    if not os.path.exists(path):
        raise ValueError(f"path to xmodel does not exist, {path}")
    overlay.load_model(path)
    
    dpu = overlay.runner

    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()

    shapeIn = tuple(inputTensors[0].dims)
    shapeOut = tuple(outputTensors[0].dims)

    output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
    input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]

    alive = True
    cam = open_video()

    while cam.isOpened() and alive:
        start_time = time.time()

        frame = get_frame(cam)

        gesture = ""
        prob = ""

        if frame is not None:            
            frame_fixed = frame.reshape(1, 3, 1080, 1920)
            input_data[0] = frame_fixed
            job_id = dpu.execute_async(input_data, output_data)
            dpu.wait(job_id)
            y = output_data[0]


        end_time = time.time()
        dt = end_time - start_time
        alive = display_image(frame, gesture, prob, dt)

        send_message()


if __name__ == '__main__':
    main()
