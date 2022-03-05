from PIL import Image
import numpy as np
import os
import time
from pynq_dpu import DpuOverlay

from app_kv260 import process_output, preprocess_frame

if __name__ == '__main__':
    im_path = 'test_image_fist.jpg'
    im = Image.open(im_path)
    arr = np.array(im)
    arr = np.moveaxis(arr, -1, 0)  # move channels to front for pytorch

    # set up model
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

    start_time = time.time()
    x = preprocess_frame(arr)

    input_data[0] = x
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    y = output_data[0]
    gesture, prob = process_output(y, detection_threshold=.05)

    end_time = time.time()
    dt = end_time - start_time

    print(f"Time: {dt}")
    print(f"Gesture: {gesture}, Prob: {prob}")