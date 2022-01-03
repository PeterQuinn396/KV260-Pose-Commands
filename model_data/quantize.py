import torch
from common import device
from model import GestureClassifyModel, get_dataloader, test
import argparse

from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from resnet import resnet101

"""
To be run inside the Vitis AI docker, after running `conda activate vitis-ai-pytorch`
"""

def quantize(model, quant_mode, batch_size):

    if quant_mode == 'test':
        batch_size = 1
        rand_in = torch.randn([1, 3, 224, 224])
    else:
        rand_in = torch.randn([batch_size, 3, 224, 224])

    # force to merge BN with CONV for better quantization accuracy
    # magic value that gets parsed by vitis?
    # optimize = 1


    quantizer = torch_quantizer(quant_mode, model, (rand_in), device=device)
    quant_model = quantizer.quant_model

    # make custom test fn, that takes a torch data loader and a loss_fn
    dataloader = get_dataloader('dataset/original_frames', batch_size=batch_size)
    acc1_gen, loss_gen, count = test(quant_model, dataloader, max_samples=10)
    print(f'Acc: {acc1_gen}, loss: {loss_gen}, count {count}')

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

    batch_size = 4

    #model = GestureClassifyModel("asl_resnext_old_fmt.pth")
    model = resnet101(pretrained=False)
    model.fc = model.fc = torch.nn.Linear(model.fc.in_features, 24)
    model.to(device)

    quantize(model, args.quant_mode, batch_size)

if __name__ == "__main__":
    main()
