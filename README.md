# KV260-Pose-Commands




Dependencies:
```text
opencv-python
torch
mediapipe 
```

A .whl file compiled for mediapipe for the KV260 is provided, as there is no pip installation for aarch64 devices. 

For issues with quantization or compilation, see the [tips and troubleshooting](model_data/readme.md)


## Peformance notes

I did some experiments with a torchvision ResNet18 model. Here are some comparisons.

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
(1200$)

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
 
