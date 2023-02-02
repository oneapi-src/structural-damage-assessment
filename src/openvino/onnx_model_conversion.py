#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# flake8: noqa = E501
# pylint: disable=missing-module-docstring

import argparse
import torch
import torch.onnx
from unet import UNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument('-m',
                        '--fp32modelpath',
                        type=str,
                        required=True,
                        default="",
                        help='fp32 Model Path')
    parser.add_argument('-output',
                        '--onnxmodelpath',
                        type=str,
                        required=True,
                        default="",
                        help='onnx Model Path')

    FLAGS = parser.parse_args()
    PATH = FLAGS.fp32modelpath
    onnx_model_savedpath = FLAGS.onnxmodelpath

# Instantiate Model
segmentation_model = UNet(n_channels=3, n_classes=4, bilinear=False)
checkpoint = torch.load(PATH)
segmentation_model.load_state_dict(checkpoint['model_state_dict'])


print("Set the model in eval mode...")
segmentation_model.eval()


batch_size, channels, height, width = 1, 3, 512, 512  # Dummy input initialization
inputs = torch.randn((batch_size, channels, height, width))
onnx_model_savedpath = onnx_model_savedpath + '/unet_model.onnx' 
torch.onnx.export( segmentation_model, inputs, onnx_model_savedpath,opset_version=11) # Save the model
