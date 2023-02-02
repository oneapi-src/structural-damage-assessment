#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
 model inference.
"""

# pylint: disable=E1101, C0103, E0401, W0622 , R0402 , R0205 , W0621 , C0411 , R1727 , W0311 , W0611
# flake8: noqa = E501


from __future__ import print_function
import argparse
import os
import random
import time
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms.functional as TF
from unet import UNet
import numpy as np
from tqdm import tqdm


def dice_index(input, target):
    '''
    Calcualtes the Dice Coefficeint for the 2 specified images

    Arguments:
    pred  --  The predicted image by the neural network
    target  --  The ground truth image/segmented image

    Returns
    coeff.item()  --  The Dice Co-efficient(float) for the predicted image and the ground-truth image
    '''
    smooth = 1.  # Factor to prevent NaN and maintain smoothness
    pred = nn.Softmax2d()(input)  # Apply Softmax since the network outputs the logits
    target = target.unsqueeze(1)
    target = torch.cat((target==0, target==1, target==2, target==3), dim=1).type(torch.float)  # Accomodate all 3 channels
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2. * intersection + smooth)/(union + smooth)).mean().item()


def IoU(input, target):
    '''
    Calcualtes the IoU Coefficeint for the 2 specified images

    Arguments:
    pred  --  The predicted image by the neural network
    target  --  The ground truth image/segmented image

    Returns
    iou.item()  --  The IoU Co-efficient(float) for the predicted image and the ground-truth image
    '''
    smooth = 1.  # Factor to prevent NaN and maintain smoothness
    pred = nn.Softmax2d()(input)  # Apply Softmax since the network outputs the logits
    target = target.unsqueeze(1)
    target = torch.cat((target==0, target==1, target==2, target==3), dim=1).type(torch.float)  # Accomodate all 3 channels
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
    return ((intersection + smooth)/(union + smooth)).mean().item()


# # Data Loader class
class SegmentationDatasetLoader(object):
    ''' Class '''
    def __init__(self, img_root_dir, gt_root_dir, train = False):
        '''
        Initialize the Dataloader

        Arguments:
        img_root_dir  --  Directory containing the input image files
        gt_root_dir   --  Directory containing the output  files
        train  --  Variable tto differentiate between traning and test/val for data augmentation and transforms

        Returns:
        None
        '''
        self.train = train
        if self.train:
            idx_threshold = 1000
        else:
            idx_threshold = 200
        self.img_root_dir = img_root_dir
        self.gt_root_dir = gt_root_dir
        self.img_list = os.listdir(img_root_dir)[:idx_threshold]
        self.gt_list  = os.listdir(gt_root_dir)[:idx_threshold]
        self.img_list.sort()
        self.gt_list.sort()


    def __getitem__(self, idx):
        '''
        Based on the input index, reads a filen and the corresponding target and outputs both as processed tensors to the net

        Arguments:
        idx  --  The index of the dataframe row to be loaded

        Returns:
        img  --  The image tensor of dimension [batch,channels,height,width]
        gt   --  The target for the image loaded in a tensor format [batch,1,height,width]
        '''
        dim = 512
        img = Image.open(os.path.join(self.img_root_dir, self.img_list[idx]))
        gt = cv2.imread(os.path.join(self.gt_root_dir, self.gt_list[idx]))
        # Make the white color as green for one hot encoding
        gt = gt[:,:,[2,1,0]]
        gt[:,:,0] = gt[:,:,0] - gt[:,:,1]
        gt[:,:,2] = gt[:,:,2] - gt[:,:,1]
        gt = Image.fromarray(gt, 'RGB')
        # Resize to a power of 2
        img = TF.resize(img, (dim, dim))
        gt = TF.resize(gt, (dim, dim))
        # Transform only for the training phase
        if self.train:
            if random.random() > 0.5:
                img = TF.vflip(img)
                gt  = TF.vflip(gt)
            if random.random() > 0.5:
                img = TF.hflip(img)
                gt  = TF.hflip(gt)
            if random.random() > 0.5:
                angle = random.randint(0, 45)
                img = TF.rotate(img, angle)
                gt  = TF.rotate(gt,  angle, fill=(0,255,0)) # Fill backgound as green
        # Convert to a tensor from a PIL Image
        img = TF.to_tensor(img)
        gt  = TF.to_tensor(gt)
        gt  = torch.argmax(gt, dim = 0)
        return img.type(torch.float), gt.type(torch.long)

    def __len__(self):
        '''
        Calculate the number of file/data-points in the directory

        Arguments:
        None

        Returns:
        Number of files
        '''
        return len(self.img_list)


def get_args():
    ''' Args '''
    parser = argparse.ArgumentParser(description='Inference on  test images with FP32/INT8 model ')
    parser.add_argument('--intel', '-i', metavar='I', type=int, default=0, help='Intel Optimizations for pytorch, default value 0')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--save_model_path','-m',type=str,required=True, default=None, help='give the directory of the trained checkpoint.')
    parser.add_argument('--data_path','-d',type=str,required=True, default=None, help='give the directory of the test data folder.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    CUDA = torch.cuda.is_available()
    print("CUDA :: ", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")

    BATCH = args.batch_size
    INTEL = args.intel
    MODEL_PATH=args.save_model_path
    DATAPATH=args.data_path
    # Loading test data
    test_label_directory =  DATAPATH + "/targets"
    test_image_directory =  DATAPATH +"/images"

    # Model Loaded Path
    print("Defined model path is ", MODEL_PATH)

    if '.tar' or '.pt' in MODEL_PATH :
        # Instantiate Model
        segmentation_model = UNet(n_channels=3, n_classes=4, bilinear=False)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if  '.tar' in MODEL_PATH:
                segmentation_model.load_state_dict(checkpoint['model_state_dict'])
        elif '.pt' in MODEL_PATH:
                from neural_compressor.utils.pytorch import load
                segmentation_model=load(MODEL_PATH,segmentation_model) 
        

        print("Loaded Weights for Inferencing...")
        segmentation_model.eval()
        segmentation_model=segmentation_model.to(memory_format=torch.channels_last)
 
        # For IPEX
        if INTEL:
                import intel_extension_for_pytorch as ipex
                segmentation_model = ipex.optimize(segmentation_model)
                print("IPEX Optimizations Enabled")
    else:
        print("Model format not supported")
    

    loader = SegmentationDatasetLoader(test_image_directory, test_label_directory, train=False)
    test_iter = torch.utils.data.DataLoader(loader, batch_size=BATCH)

    dice_test = 0
    iou_test  = 0
    acc_test  = 0
    test_iterations = len(test_iter)
    
    print("Model_warmup_initiated")

    with torch.no_grad():
        for _ in range(2):
            for img_test, label_test in test_iter:
                img_test, label_test = img_test.to(device), label_test.to(device)
                img_test = img_test.to(memory_format=torch.channels_last)
                output_test = segmentation_model(img_test)
        print("Warm up completed for this inference run " )
        # Test main Loop for inference 
        with torch.no_grad():
                for _ in range(10):
                        for img_test, label_test in test_iter:
                                img_test, label_test = img_test.to(device), label_test.to(device)
                                #print(img_test.shape)
                                img_test = img_test.to(memory_format=torch.channels_last)
                                start_time = time.time()
                                output_test = segmentation_model(img_test)
                                print("Time Taken for Inferencing ", BATCH, " Images is ==>",time.time()-start_time )
                                


      