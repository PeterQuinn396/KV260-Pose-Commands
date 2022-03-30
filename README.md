# KV260-Pose-Commands

This is the code associated with an article I wrote over on hackster.io.

The app runs at about 3 FPS, which is sufficient for interacting with the menu UI in realtime. I think most of the run time is from openCV grabbing frames and mediapipe preprocessing the data for the model that is running on the KV260 DPU.

## Main highlights

1. The `model_data/quantize.py` script
    - A helper script to quantize the weights of a model 
2. `app_kv260.py` 
    - A python script to run the app on the KV260
    - The app detects hand gestures on the KV260 using a USB webcam and sends commands to a FireTV device
3. `controllers/firetv_controller.py`
    - A python script to send commands to a FireTV


### Additional features
1. A 3D printable stand that I found on the internet.
   - https://3dmixers.com/m/122358-kria-kv260-3d-printed-stand-production-no-supports

2. A `.whl` file compiled for mediapipe for the KV260 is provided, as there is no pip installation for aarch64 devices.

3. A few small scripts prefixed with `test_` to test the different parts of the app.

## Dependencies
```text
opencv-python
torch
mediapipe 
```

A `.whl` file compiled for mediapipe for the KV260 is provided, as there is no pip installation for aarch64 devices. 

# Troubleshooting

For issues with quantization or compilation, see the [tips and troubleshooting](model_data/readme.md)


# Peformance notes
I did some experiments with a torchvision ResNet18 model. Here are some comparisons.

The KV260 is much faster than a CPU computation. The KV260 performs quite well when compare to a high performance GPU, especially when you consider the cost of the hardware (~300$ for the KV260 vs well over 1000$ for the Tesla K80 and the rest of the computer).

### KV260
Tested in Jupyter notebooks, using %%timeit
```python=
%%timeit
input_data[0] = x # [1,3,1080,1080]
job_id = dpu.execute_async(input_data, output_data)
dpu.wait(job_id)
y = output_data[0]

# 123 ms ± 72 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

### Cuda
Tested on Google Colab, with a GPU runtime with a Tesla K80 (12GB)


```python=
%%timeit
y = model(test_input) # [1,3,1080,1080], after to device

# 100 loops, best of 5: 84 ms per loop
```

### CPU
Tested on Google Colab, Intel(R) Xeon(R) CPU @ 2.20GHz

```python=
%%timeit
y = model(test_input) # [1,3,1080,1080], after to device
# 1 loop, best of 5: 1.92 s per loop 
```
 
