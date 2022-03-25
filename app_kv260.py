import os.path
import argparse
import numpy as np
import time
import cv2
from controllers.firetv_controller import FireTVController
from typing import Tuple

from pynq_dpu import DpuOverlay
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hand_detector = mp_hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5, max_num_hands=1)

CATEGORIES = ['up', 'down', 'left', 'right', 'fist', 'palm']


def open_video(cam_id=0) -> cv2.VideoCapture:
    """Open the camera and return it.

    Args:
        cam_id: id of the camera to open.

        This function is set up to use the first camera on the system. It also sets the capture resolution to 1080x1920.
        Note that these settings are not guaranteed to work with all cameras.

    Returns:
        cv2.VideoCapture object
    """
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    buf_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    print(f"Buffer size: {buf_size}")  # make sure there is no build up of frames from slow processing
    return cap


def get_frame(cam) -> np.ndarray:
    """Get a frame from the camera and return it.

    Args:
        cam: opencv camera object to get the frame from.

    Returns:
        RGB np.ndarray of the frame.
    """
    success, image = cam.read()
    if not success:
        print("Got empty camera frame")
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_3d_points(results) -> np.ndarray:
    """Process results from mediapipe to a flat vector, formatted for the ML model input

    If you see soemthing like 'results.multi_hand_world_landmarks[0].landmark key/index/attribute does not exist`,
    you might be using an older version of mediapipe which doesn't have multi_hand_world_landmarks

    Args:
        results: results from

    Returns:
        [1,63] array

    """
    arr = []
    for pt in results.multi_hand_world_landmarks[0].landmark:
        coords = [pt.x, pt.y, pt.z]
        arr.append(coords)
    arr = np.array(arr).reshape(3, 1, 21).astype(np.float32).flatten().squeeze()
    return arr


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def process_output(outputTensor: np.ndarray, detection_threshold: float = .05) -> Tuple[str, float]:
    """Process the output tensor to determine which gesture it identified.

    Args:
        outputTensor: tensor from the ML model.
        detection_threshold: minimum level of confidence for model to detect a gesture.

    Returns:
        gesture: string name of gesture, if detection threshold was met for at least one element,
        None otherwise
        max_prob: the probability of the gesture being the one identified.

    """
    out = outputTensor.squeeze()
    probs = softmax(out)
    max_prob = np.max(probs)
    if max_prob > detection_threshold:
        gesture = CATEGORIES[np.argmax(probs)]
    else:
        gesture = None
    return gesture, max_prob


def send_message(firetv_controller: FireTVController, gesture: str):
    """Call the send_command function with the appropriate command for the gesture.

    This function maps the gestures to the tv functions we want to control.

    Args:
        firetv_controller:
        gesture:

    """
    if gesture == 'up':
        firetv_controller.send_command('up')
    elif gesture == 'down':
        firetv_controller.send_command('down')
    elif gesture == 'left':
        firetv_controller.send_command('left')
    elif gesture == 'right':
        firetv_controller.send_command('right')
    elif gesture == 'palm':
        firetv_controller.send_command('select')
    elif gesture == 'fist':
        firetv_controller.send_command('back')


def display_image(image: np.ndarray, gesture: str, probability: float, time: float) -> bool:
    """Write some text on the image and display it using opencv

    Args:
        image: rgb image
        gesture: name of gesture, to write on display
        probability:  probability of gesture, to write on display
        time: Time between frames. Used to draw a frame rate

    Returns:

    """
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


def main(display=False):
    # Set up the DPU by loading our model and allocating memory for the input and output
    overlay = DpuOverlay("dpu.bit")

    path = '/home/ubuntu/KV260-Pose-Commands/model_data/custom_dataset/simple_mlp_kv260.xmodel'

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

    firetv_controller = FireTVController()  # Insert your own controller code here
    firetv_controller.add_device("192.168.2.138")  # Change to match your device IP address

    # Open up the webcam
    alive = True
    cam = open_video()

    while cam.isOpened() and alive:
        start_time = time.time()

        frame = get_frame(cam)

        if frame is None:
            continue

        im = frame[400:800, 500:1500, :]  # clip frame to ROI specifically for my hardware setup

        results = hand_detector.process(im)  # detect hands in image

        if results.multi_hand_landmarks is None:
            continue

        x = get_3d_points(results)

        input_data[0] = x
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)
        y = output_data[0]
        gesture, prob = process_output(y, detection_threshold=.2)

        end_time = time.time()
        dt = end_time - start_time

        # Draw hand landmarks and display image using opencv
        if display:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(im, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            alive = display_image(im, gesture, prob, dt)

        print(f"{gesture}, Prob: {prob}, Time: {dt}")
        send_message(firetv_controller, gesture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', '-d', action='store_true',
                        help='Displays the webcam view with some debug information printed on it.')
    args = parser.parse_args()
    main(display=args.display)
