#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
 initial model training.
"""

# pylint: disable=E1101, C0103, E0401, W0622 , R0402 , R0205 ,W0621 , C0411
# flake8: noqa = E501

# # Import all Libraries

from __future__ import print_function
import os
import sys
import random
import argparse
import time
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms.functional as TF
from unet import UNet
import cv2
import numpy as np
import pathlib
from sklearn.metrics import accuracy_score


# Define function for input & Target specification
def dice_index(input, target):
    '''
    Calcualtes the Dice Coefficeint for the 2 specified images

    Arguments:
    pred  --  The predicted image by the neural network
    target  --  The ground truth image/segmented image

    Returns
    coeff.item()--The Dice Co-efficient(float) for the predicted image and the ground-truth image
    '''
    smooth = 1.  # Factor to prevent NaN and maintain smoothness
    pred = nn.Softmax2d()(input)  # Apply Softmax since the network outputs the logits
    target = target.unsqueeze(1)
    target = torch.cat((target==0, target==1, target==2, target==3), dim=1).type(torch.float)  # Accomodate all 3 channels
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
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
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + smooth)/(union + smooth)).mean().item()


# # Data Loader class Initialization

class SegmentationDatasetLoader(object):
    '''
        Initialize the class
    '''

    def __init__(self, img_root_dir, gt_root_dir, train=False):
        '''
        Initialize the Dataloader

        Arguments:
        img_root_dir  --  Directory containing the input image files
        gt_root_dir   --  Directory containing the output label files
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
        self.gt_list = os.listdir(gt_root_dir)[:idx_threshold]
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
        gt = gt[:, :, [2, 1, 0]]
        gt[:, :, 0] = gt[:, :, 0] - gt[:, :, 1]
        gt[:, :, 2] = gt[:, :, 2] - gt[:, :, 1]
        gt = Image.fromarray(gt, 'RGB')
        # Resize to a power of 2
        img = TF.resize(img, (dim, dim))
        gt = TF.resize(gt, (dim, dim))
        # Transform only for the training phase
        if self.train:
            if random.random() > 0.5:
                img = TF.vflip(img)
                gt = TF.vflip(gt)
            if random.random() > 0.5:
                img = TF.hflip(img)
                gt = TF.hflip(gt)
            if random.random() > 0.5:
                angle = random.randint(0, 45)
                img = TF.rotate(img, angle)
                gt = TF.rotate(gt, angle, fill=(0, 255, 0))  # Fill backgound as green
        #  Convert to a tensor from a PIL Image
        img = TF.to_tensor(img)
        gt = TF.to_tensor(gt)
        gt = torch.argmax(gt, dim=0)
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


#  Startup code  where execution of code should start
if __name__ == "__main__":
    # The main function body which takens the varilable number of arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=6,
                        help='batch size exampes: 6, 12')

    parser.add_argument('--dataset_file',
                        '--dataset_file',
                        type=str,
                        required=True,
                        default=None,
                        help='dataset file for training')

    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 to enable intel model save, \
                            default is 0')

    parser.add_argument('--save_model_path',
                        '--save_model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='give the directory path to save the model')

    # Holds all the arguments passed to the function
    FLAGS = parser.parse_args()
    datadir = FLAGS.dataset_file
    BATCH = FLAGS.batch_size
    PATH = pathlib.Path(FLAGS.save_model_path)
    os.makedirs(PATH, exist_ok=True)

    if FLAGS.batch_size < 0:
        print("The parameter batch size value is invalid, try with valid \
            value\n")
        sys.exit(1)
    if FLAGS.save_model_path is None:
        print("Please provide path to save the model...\n")
        sys.exit(1)
    else:
        if FLAGS.intel != 1:
            save_model_path = FLAGS.save_model_path + "/stock/"
            os.makedirs(save_model_path, exist_ok=True)
        else:
            save_model_path = FLAGS.save_model_path + "/intel/"
            os.makedirs(save_model_path, exist_ok=True)

    # Handle Exceptions for the user entries
    try:
        if not os.path.exists(FLAGS.dataset_file):
            print("Dataset file path Not Found!!")
            raise FileNotFoundError
    except FileNotFoundError:
        print("Please check the Path provided!")
        sys.exit()


    CUDA = torch.cuda.is_available()
    print("CUDA :: ", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")

    # Define of datapath
    image_directory = datadir + "/train/images"
    gt_directory = datadir + "/train/targets"

    val_image_directory = datadir + "/test/images"
    val_gt_directory = datadir + "/test/targets"

    test_label_directory = datadir + "/test/targets"
    test_image_directory = datadir + "/test/images"


    # # # # Default Hyper Parameters & Network deatils ,Optimizer
    hidden = 64
    kernel_size = 3
    padding = 1
    p1 = 0.25
    p2 = 0.5
    learning_rate = 0.0001
    resume_epoch = 0
    num_epochs = 5

    # Training Save Path
    PATH = save_model_path
    print("Train Path exists : ", os.path.exists(PATH))
    PATH += "checkpoint"

    # Testing Load Path
    TEST_PATH = PATH
    print("The loaded checkpoint path is :", TEST_PATH)


    # # Define the U-Net Generator Model & Instatiate the Model, Optimizer and Losses

    segmentation_model = UNet(n_channels=3, n_classes=4, bilinear=False)
    # Setup Optimizer
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=learning_rate)
    # Initialize Cross Entropy Loss
    criterion_cross = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1, 1], dtype=torch.float).to(device))
    segmentation_model.train()


    if FLAGS.intel == 1:

        import intel_extension_for_pytorch as ipex
        segmentation_model = segmentation_model.to(memory_format=torch.channels_last)
        segmentation_model, optimizer = ipex.optimize(segmentation_model, optimizer=optimizer)
        print("IPEX optimization enabled")


    # Instantiate the Training DataLoader and Data Iterator
    segmentation_dataset = SegmentationDatasetLoader(image_directory, gt_directory, train=True)
    segmentation_iter = torch.utils.data.DataLoader(segmentation_dataset, batch_size=BATCH, shuffle=True)

    # Instantiate the Validation Data Loader and Data Iterator
    # val_dataset = SegmentationDatasetLoader(val_image_directory, val_gt_directory)
    # val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH, shuffle=False)



    # Initiating Training
    loss_train = []
    # Get train and valid iteration size
    iterations = len(segmentation_iter)
    # Resume Check
    if resume_epoch == 0:
        print("Starting Training Loop...")
    else:
        checkpoint = torch.load(PATH + str(resume_epoch) + '.tar', map_location=device)
        segmentation_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Resuming Training Loop...")


    start_train_time = time.time()
    # Training Loop
    for epoch in range(resume_epoch, num_epochs):
        loss_epoch = 0
        dice = 0
        dice_val = 0
        iou = 0
        iou_val = 0
        acc_train = 0
        acc_val = 0
        segmentation_model.train()
        for img, label in tqdm(segmentation_iter):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            output = segmentation_model(img)
            loss = criterion_cross(output, label)
            loss_epoch += loss.item()

            # Metrics for Training
            acc_train += accuracy_score(label.clone().detach().cpu().numpy().ravel(),  np.argmax(output.clone().detach().cpu().numpy(), axis=1).ravel())
            dice += dice_index(output.clone().detach(), label.clone().detach())
            iou += IoU(output.clone().detach(), label.clone().detach())

            # Backprop
            loss.backward()
            optimizer.step()

        loss_train.append(loss_epoch)
        print("Epoch : ", epoch + 1, " Loss : ", loss_epoch, " Dice : ", dice/iterations, " IoU : ", iou/iterations, "Accuracy : ", acc_train/iterations)

        torch.save({
          'epoch': epoch+1,
           'model_state_dict': segmentation_model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'loss': loss_epoch}, PATH +'.tar')

    print("TOTAL TIME TAKEN FOR TRAINING IN SECONDS --> ", time.time()-start_train_time)


    #  Test Dataloader to get accuracy on test dataset


    loader = SegmentationDatasetLoader(test_image_directory, test_label_directory, train=False)
    test_iter = torch.utils.data.DataLoader(loader, batch_size=BATCH)
    dice_test = 0
    iou_test = 0
    acc_test = 0
    test_iterations = len(test_iter)


    # Load model state dictionary
    segmentation_model = UNet(n_channels=3, n_classes=4, bilinear=False)
    checkpoint = torch.load(PATH +'.tar', map_location=device)
    segmentation_model.load_state_dict(checkpoint['model_state_dict'])
    segmentation_model.eval()

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
