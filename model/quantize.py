import torch
from .common import device
from .asl_dataset import ASLDataset, preprocess
from .model import loss_fn, GestureClassifyModel, get_dataloader, test
import argparse

from pytorch_nndct.apis import torch_quantizer, dump_xmodel

"""
To be run inside the Vitis AI docker, after running `conda activate vitis-ai-pytorch`
"""

def quantize(model, val_loader, quant_mode):
    input = torch.randn([1, 3, 224, 224])
    quantizer = torch_quantizer(quant_mode, model, (input))
    quant_model = quantizer.quant_model

    # make custom test fn, that takes a torch data loader and a loss_fn
    acc1_gen, loss_gen = test(quant_model, val_loader)
    print(f'Acc: {acc1_gen}, loss: {loss_gen}')

    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False)


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--quant_mode', choices=['test', 'calib'])
    parser.add_argument('--deploy', action='store_true')
    parser.add_argument('--subset_length', default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()


    model = GestureClassifyModel("asl_resnext.pth")
    model.to(device)
    dataloader = get_dataloader('dataset/original_frames')

    quantize(model, dataloader, args.quant_mode)

if __name__ == "__main__":
    main()
