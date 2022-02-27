import os.path

import numpy as np
import time
import cv2
from typing import List

from pynq_dpu import DpuOverlay

CATEGORIES = ['up', 'down', 'left', 'right', 'fist', 'palm']


def open_video(cam_id=0):
    cap = cv2.VideoCapture(cam_id + cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: cannot open camera")
        exit(-1)
    # grab test frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit(-1)

    # setup proper backend and codec to allow for 1920x1080 frames
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Cam resolution: {width}, {height}")

    return cap


def get_frame(cam):
    success, image = cam.read()
    if not success:
        print("Got empty camera frame")
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    return image


def preprocess_frame(frame):
    mid = 1920 // 2
    ext = 1080 // 2
    x = frame / 255.
    x = np.moveaxis(x, -1, 0)
    x = x[..., mid - ext:mid + ext]
    x = np.reshape(x, (1, 3, 1080, 1080))
    return x.astype(np.float32)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def process_output(outputTensor, detection_threshold=.75):
    out = outputTensor.squeeze()
    probs = softmax(out)
    max_prob = np.max(probs)
    if max_prob > detection_threshold:
        gesture = CATEGORIES[np.argmax(probs)]
    else:
        gesture = None
    return gesture, max_prob


def send_message():
    pass


def display_image(image, gesture, probability, time) -> bool:


    # input is rgb
    im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4))

    if time > 0:  # avoid a crash from a bad time
        cv2.putText(im, f"FPS: {1 / time:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv2.putText(im, f"Gesture: {gesture}, Prob: {probability}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv2.imshow("Image", im)
    alive = True
    if cv2.waitKey(1) & 0xFF == 27:
        alive = False
    return alive


def main():
    overlay = DpuOverlay("dpu.bit")

    path = '/home/ubuntu/KV260-Pose-Commands/model_data/custom_dataset/resnet_square.xmodel'

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
            x = preprocess_frame(frame)
            input_data[0] = x
            job_id = dpu.execute_async(input_data, output_data)
            dpu.wait(job_id)
            y = output_data[0]
            gesture, prob = process_output(y, detection_threshold=.75)

        end_time = time.time()
        dt = end_time - start_time
        alive = display_image(frame, gesture, prob, dt)
        print(f"{gesture}, {prob}, {dt}")
        send_message()


if __name__ == '__main__':
    main()
