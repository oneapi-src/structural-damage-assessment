#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable = R0402 , W0404 , W0622 , C0115 , R0205 , W0404 , C0411 , C0412 , W0611 , E0401, W0311, E1102 , E0213
# flake8: noqa = E501

"""
INT8 model conversion
"""

from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms.functional as TF
from unet import UNet
import time
import torch
from tqdm import tqdm
##from utils.dataloaders import create_dataloader
#from utils.general import check_img_size, colorstr, check_yaml, check_dataset
#from models.common import DetectMultiBackend
#import intel_extension_for_pytorch as ipex
from neural_compressor.experimental import Quantization, common


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


class Dataset:
    
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


     def evalution(img_root_dir, gt_root_dir, train = False):
          
       
          #import pdb;pdb.set_trace()
          loader = Dataset(test_image_directory, test_label_directory, train=False)
          test_iter = torch.utils.data.DataLoader(loader, batch_size=BATCH)
          dice_test = 0
          iou_test = 0
          acc_test = 0
          test_iterations = len(test_iter)


          segmentation_model = UNet(n_channels=3, n_classes=4, bilinear=False)

          checkpoint = torch.load(TEST_PATH, map_location=device)
          segmentation_model.load_state_dict(checkpoint['model_state_dict'])
          print("Loaded model Accuracy")
          segmentation_model.eval()
          segmentation_model=segmentation_model.to(memory_format=torch.channels_last)
        
          # Test Loop
          with torch.no_grad():
               for img_test, label_test in tqdm(test_iter):
                    img_test, label_test = img_test.to(device), label_test.to(device)
                    output_test = segmentation_model(img_test)
                    # Metrics for the Testing
                    acc_test += accuracy_score(label_test.clone().detach().cpu().numpy().ravel(),  np.argmax(output_test.clone().detach().cpu().numpy(), axis=1).ravel())
                    dice_test += dice_index(output_test.clone().detach(), label_test.clone().detach())
                    iou_test  +=  IoU(output_test.clone().detach(), label_test.clone().detach())
                    
          print("Test Dice : ", dice_test/test_iterations, " IoU : ", iou_test/test_iterations, " Acc : ", acc_test/test_iterations)
          x=acc_test/test_iterations          
          return x

def get_args():
    parser = argparse.ArgumentParser(description='Quantized the UNet')
    parser.add_argument('-i', '--intelflag', type=int, default=0, help='For enabling IPEX Optimizations value  of i will be 1 but INC it has to be 0')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=10, help='Batch size')
    parser.add_argument('-o','--outpath',type=str,required=False,default='./inc_compressed_model/output',help='absolute path to save quantized model. By default it ''will be saved in "./inc_compressed_model/output" folder')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default='./deploy.yaml',
                        help='Yaml file for quantizing model, default is "./deploy.yaml"')

    parser.add_argument('--save_model_path',
                        '--save_model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='give the directory path to save the Quantized model')
   
   
    return parser.parse_args()

    
if __name__ == '__main__':
    args = get_args()

    CUDA = torch.cuda.is_available()
    print("CUDA :: ", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")

    # Dataloader
    BATCH = args.batch_size
    flag = args.intelflag 
    config_path = args.config
    out_path =  args.outpath
    TEST_PATH = args.save_model_path
    

    test_label_directory = "./data/test/targets"
    test_image_directory = "./data/test/images"

    # Testing Load Path
    print("Loaded_FP32model_path_is",TEST_PATH)

    # Instantiate Model
    segmentation_model = UNet(n_channels=3, n_classes=4, bilinear=False)
    checkpoint = torch.load(TEST_PATH, map_location=device)
    segmentation_model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded Weights for Inferencing...")
    segmentation_model.eval()
    segmentation_model=segmentation_model.to(memory_format=torch.channels_last)

    if flag ==1:
        import intel_extension_for_pytorch as ipex
        segmentation_model = ipex.optimize(segmentation_model)
        print("IPEX Optimizations Enabled")
    else :
        print("Quantization will be done without IPEX Optimizations Enabled")
    
    # Quantization
    quantizer = Quantization(config_path)
    quantizer.model = segmentation_model
    dataset = Dataset(test_image_directory, test_label_directory, train=False)
    quantizer.calib_dataloader = common.DataLoader(dataset)
    quantizer.eval_func =  dataset.evalution
    q_model = quantizer.fit()
    q_model.save(out_path)
