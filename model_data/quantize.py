import sys

import torch
from common import device

import argparse

from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from common import test_vitis_compatible
from custom_dataset.model import (get_model, get_dataloader, CATEGORIES, GestureDatasetFromFile, get_dataloader_files,
    get_pickle_dataset)

"""
To be run inside the Vitis AI docker, after running `conda activate vitis-ai-pytorch`
"""


def load_model():
    print("Loading model...")
    m = get_model()

    # print(m)
    m.load_state_dict(torch.load("custom_dataset/gesture_resnet18_custom_dataset_old_fmt.pt", map_location=device))

    m.eval()
    m.to(device)
    print('Done')

    return m


def test(model, dataloader):
    model.eval()
    acc = 0
    with torch.no_grad():
        for x, y_ref in dataloader:
            print("to device")
            x.to(device)
            y_ref.to(device)
            print('running model')
            y_pred = model(x)
            print('getting ans')
            _, predicted = y_pred.max(dim=1)
            correct = (predicted == y_ref)
            print('acumulate')
            acc += 1.0 * correct.sum().item() / y_ref.shape[0]
            print('loop')
    acc /= len(dataloader)
    return acc


def test_dataset(model, dataset):
    model.eval()
    acc = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            print(f'{i+1}/{len(dataset)}')
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
    if quant_mode == 'test':
        batch_size = 1
    else:
        batch_size = 1

    rand_size = [batch_size, 3, 1080, 1920]
    rand_in = torch.randn(rand_size)
    print(f"Rand in size: {rand_in.size()}")
    if not test_vitis_compatible(model, rand_size):
        print('model failed jit test')
        exit()

    # force to merge BN with CONV for better quantization accuracy
    # magic value that gets parsed by vitis?
    # optimize = 1
    model.eval()
    quantizer = torch_quantizer(quant_mode, model, (rand_in), device=device)  # qat_proc=True

    quant_model = quantizer.quant_model

    # make custom test fn, that takes a torch data loader and a loss_fn

    if quant_mode == 'calib':
        print("Getting data loader...")

        dataset = get_pickle_dataset('custom_dataset/custom_dataset_test.pckl')
        print(len(dataset))

        print("Got loader")
        print("Running test set...")

        acc1_gen = test_dataset(quant_model, dataset)

        print(f'Acc: {acc1_gen}')  # , loss: {loss_gen}, count {count}')

        print("Running export_quant_config...")
        quantizer.export_quant_config()

    if quant_mode == 'test':
        y = quant_model(rand_in)

        print("Running export_xmodel...")
        quantizer.export_xmodel(deploy_check=True)


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--quant_mode', choices=['test', 'calib'],
                        help='Run calib mode before test mode')
    # parser.add_argument('--deploy', action='store_true')
    # parser.add_argument('--subset_length', default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    model = load_model()
    model.to(device)

    quantize(model, args.quant_mode)


if __name__ == "__main__":
    main()
