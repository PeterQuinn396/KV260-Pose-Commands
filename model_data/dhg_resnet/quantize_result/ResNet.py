# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ResNet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Conv2d[conv1]/input.2
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/ReLU[relu]/2680
        self.module_4 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False) #ResNet::ResNet/MaxPool2d[maxpool]/input.4
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]/input.5
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/input.7
        self.module_8 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]/input.8
        self.module_10 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/input.9
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/input.10
        self.module_12 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]/input.11
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/input.13
        self.module_15 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/input.14
        self.module_17 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/input.15
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/input.16
        self.module_19 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]/input.17
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/input.19
        self.module_22 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]/input.20
        self.module_24 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.21
        self.module_26 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/input.22
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/input.23
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]/input.24
        self.module_30 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/input.26
        self.module_31 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]/input.27
        self.module_33 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/input.28
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/input.29
        self.module_35 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]/input.30
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/input.32
        self.module_38 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]/input.33
        self.module_40 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.34
        self.module_42 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/input.35
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/input.36
        self.module_44 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]/input.37
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/input.39
        self.module_47 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]/input.40
        self.module_49 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/input.41
        self.module_50 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/input.42
        self.module_51 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]/input.43
        self.module_53 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/input.45
        self.module_54 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]/input.46
        self.module_56 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.47
        self.module_58 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/input.48
        self.module_59 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/input.49
        self.module_60 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/input.50
        self.module_62 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/input.52
        self.module_63 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]/input.53
        self.module_65 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/input.54
        self.module_66 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/input.55
        self.module_67 = py_nndct.nn.AdaptiveAvgPool2d(output_size=1) #ResNet::ResNet/AdaptiveAvgPool2d[avgpool]/3217
        self.module_68 = py_nndct.nn.Module('flatten') #ResNet::ResNet/input
        self.module_69 = py_nndct.nn.Linear(in_features=512, out_features=14, bias=True) #ResNet::ResNet/Linear[fc]/3224

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_5 = self.module_5(output_module_0)
        output_module_5 = self.module_7(output_module_5)
        output_module_5 = self.module_8(output_module_5)
        output_module_5 = self.module_10(input=output_module_5, other=output_module_0, alpha=1)
        output_module_5 = self.module_11(output_module_5)
        output_module_12 = self.module_12(output_module_5)
        output_module_12 = self.module_14(output_module_12)
        output_module_12 = self.module_15(output_module_12)
        output_module_12 = self.module_17(input=output_module_12, other=output_module_5, alpha=1)
        output_module_12 = self.module_18(output_module_12)
        output_module_19 = self.module_19(output_module_12)
        output_module_19 = self.module_21(output_module_19)
        output_module_19 = self.module_22(output_module_19)
        output_module_24 = self.module_24(output_module_12)
        output_module_19 = self.module_26(input=output_module_19, other=output_module_24, alpha=1)
        output_module_19 = self.module_27(output_module_19)
        output_module_28 = self.module_28(output_module_19)
        output_module_28 = self.module_30(output_module_28)
        output_module_28 = self.module_31(output_module_28)
        output_module_28 = self.module_33(input=output_module_28, other=output_module_19, alpha=1)
        output_module_28 = self.module_34(output_module_28)
        output_module_35 = self.module_35(output_module_28)
        output_module_35 = self.module_37(output_module_35)
        output_module_35 = self.module_38(output_module_35)
        output_module_40 = self.module_40(output_module_28)
        output_module_35 = self.module_42(input=output_module_35, other=output_module_40, alpha=1)
        output_module_35 = self.module_43(output_module_35)
        output_module_44 = self.module_44(output_module_35)
        output_module_44 = self.module_46(output_module_44)
        output_module_44 = self.module_47(output_module_44)
        output_module_44 = self.module_49(input=output_module_44, other=output_module_35, alpha=1)
        output_module_44 = self.module_50(output_module_44)
        output_module_51 = self.module_51(output_module_44)
        output_module_51 = self.module_53(output_module_51)
        output_module_51 = self.module_54(output_module_51)
        output_module_56 = self.module_56(output_module_44)
        output_module_51 = self.module_58(input=output_module_51, other=output_module_56, alpha=1)
        output_module_51 = self.module_59(output_module_51)
        output_module_60 = self.module_60(output_module_51)
        output_module_60 = self.module_62(output_module_60)
        output_module_60 = self.module_63(output_module_60)
        output_module_60 = self.module_65(input=output_module_60, other=output_module_51, alpha=1)
        output_module_60 = self.module_66(output_module_60)
        output_module_60 = self.module_67(output_module_60)
        output_module_60 = self.module_68(input=output_module_60, start_dim=1, end_dim=3)
        output_module_60 = self.module_69(output_module_60)
        return output_module_60
