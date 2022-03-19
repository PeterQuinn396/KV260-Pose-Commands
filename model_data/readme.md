

# Tips 

### Test your model with `torch.jit.trace` function to ensure its compatible with Vitis-AI

I provided a helper function for this in `common.py` [here](common.py)

### Make sure the DPU subgraph number is 1 when you quantize the model.

This is necessary to work with PYNQ DPU overlay. See notes in the troubleshooting for fixing this. 

# Troubleshooting

Here some notes about issues I had while working on this project, and how I solved them. Hopefully they save others some time in the future. 

## When running in the docker, the quantize script keeps stopping and just saying "Killed"

I encountered this twice. 

1. If you are using a VM to run the docker (ex. WSL2 in a Windows machine), your model/dataset might be trying to take up more memory than your system has allocated to the VM.
- Use a smaller model / dataset
- Skip testing the accuracy of the quantized model (i.e. don't load the big dataset and run inference)

2. You might be using the incorrect version of the vitis-ai docker. If you want to run just using CPU, you must use the special cpu docker.
- CPU: https://hub.docker.com/r/xilinx/vitis-ai-cpu/tags
- GPU: https://hub.docker.com/r/xilinx/vitis-ai



## The quantization crashes with non-descriptive error

With pytorch, there are many limitations with respect to what operations the vitis compiler can handle. If the quatization workflow encounters an unfamiliar operation, it typically crashes with an error that is not super descriptive. This seems to have gotten a bit better with Vitis-AI 2.0, but it's still not so great.

Inspecting the `<your_model_name>.py` file in `quantize_result` can also give some insights as to where there might be problems. Anything that gets broken out into a function with a string argument is suspicious.

Vitis-AI is only capable of quantizing a subset of pytorch operations. If your model contains any unsupported operations, you will have issues.

See page 110 in these [docs](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_4/ug1414-vitis-ai.pdf) for the list of PyTorch supported operators.

Avoid anything not on here. You also have to avoid any sort of indexing or looping in the `forward` function of your model.

Some examples of unsupported operations:

- Conv1D
- Some aspects of Vitis don't seem to support the `+` or `-` operations on tensors, use `torch.add()` and `torch.subtract()` instead
- Anything with a kernel size > 8, or non-square kernels
- The flatten operation seems like it might do weird stuff sometimes, replace it with reshape

## I get an assertion error about DPU subgraph length not being equal to 1 when trying to load my model into the DPU overlay on the KV260

The Vitis-AI docker broke your model up into multiple subgraphs during the compilation process. This happens when it encounters an operation in your model that it is not capable of properly compiling. 

To determine which operation is the problem generate an svg of your compiled model with the `xir svg <your_model>.xmodel model_graph.svg` command from inside the docker. You can inspect this to find the guilty operation. 





## My model is getting split during quantization at the `AdaptiveMaxPool2d` step

Vitis-AI only supports square kernals up to size 8x8. Your model has likely adaptively chosen a non-square or larger kernal based on the input. 


## When I try to run my compiled model on the KV260, I get `bad any cast` when trying to load it.

You might have a mismatch between the version of Vitis-AI you used to compile the model (i.e. the docker version) and the version of Vitis-AI tools installed on the KV260 (if you are using PYNQ, it should be Vitis-AI 1.4).

## Docker/vmmem is using a ton of memory and lagging computer
If you experience a lot of lag when running the docker, and task manager reports a ton of memory being used by `vmmem`, you should add/edit `C:\Users\<user_name>\.wslconfig` with something like:

``` bash
[wsl2]
memory=4GB
processors=4
```

Run PowerShell in admin mode, with command:
```bash
Restart-Service LxssManager 
```

You should then get a notification from the docker app after a few seconds prompting you to restart the service, which you should accept.

Alternatively, just restart your computer, which should restart all that for you.



