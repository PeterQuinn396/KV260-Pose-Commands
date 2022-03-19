"""
This is a script to quantize a pytorch float model to a xmodel, ready for compilation.
To be run inside the Vitis AI docker, after running `conda activate vitis-ai-pytorch`

This script is based on the example:
https://github.com/Xilinx/Vitis-AI-Tutorials/blob/master/Design_Tutorials/09-mnist_pyt/files/quantize.py
"""

import torch

from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from common import test_vitis_compatible

from custom_dataset.model_simple import get_model, get_dataset
from common import device


def load_model():
    """A function that returns a pytorch model.

    This function fetches the definition of the model using get_model, and then loads
    in the pretrained weights.

    Returns:
        pretrained model to quantize

    """
    print("Loading model...")
    m = get_model()
    m.load_state_dict(torch.load("custom_dataset/simple_mlp.pt", map_location=device))

    m.eval()
    m.to(device)
    print('Done')

    return m


def test_dataset(model, dataset):
    """An example function for testing the accuracy of the quantized model.

    The quantization process can degrade the performance for some model significantly.

    Args:
        model: the quantized model
        dataset: a pytorch dataset to iterate through

    Returns:

    """
    model.eval()
    acc = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            print(f'{i + 1}/{len(dataset)}')
            x, y_ref = dataset[i]
            x.to(device)
            y_ref.to(device)
            y_pred = model(x.unsqueeze(0))
            _, predicted = y_pred.max(dim=1)
            print(y_ref)
            print(predicted)
            correct = 1 if predicted == y_ref else 0
            print(correct)
            acc += correct
    acc /= len(dataset)
    return acc


def quantize(model, quant_mode):
    """Main function for quantizing the model

    Args:
        model: Pytorch float model.
        quant_mode: either 'test' or 'calib'
            Required setting for the torch_quantizer function

    Returns:

    """

    batch_size = 1
    rand_size = [batch_size, 63]
    rand_in = torch.randn(rand_size)
    print(f"Rand in size: {rand_in.size()}")
    if not test_vitis_compatible(model, rand_size):
        print('model failed jit test')
        exit()

    model.eval()

    quantizer = torch_quantizer(quant_mode, model, (rand_in), device=device)  # qat_proc=True

    quant_model = quantizer.quant_model

    if quant_mode == 'calib':
        print("Getting data loader...")

        # dataset = get_pickle_dataset('custom_dataset/custom_dataset_test.pckl')
        dataset = get_dataset('custom_dataset/data/')
        print(len(dataset))

        print("Got loader")
        print("Running test set...")

        # Test the data
        # acc1_gen = test_dataset(quant_model, dataset)

        # If you want to skip testing the model, you can just forward a batch of random data
        # It does seem like a forward pass must be done in order to properly set some internal
        # state of the quant model before it can be exported.
        rand_size = [4, 63]
        rand_in = torch.randn(rand_size)

        acc1_gen = 0

        quant_model(rand_in)
        print(f'Acc: {acc1_gen}')  # , loss: {loss_gen}, count {count}')

        print("Running export_quant_config...")
        quantizer.export_quant_config()

    if quant_mode == 'test':
        y = quant_model(rand_in) # You must forward the model at least once before exporting it.

        print("Running export_xmodel...")
        quantizer.export_xmodel(deploy_check=True)


def main():
    model = load_model()
    model.to(device)

    # Run both the quantize and calib steps back to back to get all everything done at once
    quantize(model, 'calib')
    quantize(model, 'test')


if __name__ == "__main__":
    main()
