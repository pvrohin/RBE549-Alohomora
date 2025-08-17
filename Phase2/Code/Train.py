#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
#import Misc.ImageUtils as iu
from Network.Network import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *



# Don't generate pyc codes
sys.dont_write_bytecode = True

device = "mps" if torch.backends.mps.is_available() else "cpu"

    
def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1)
        
        ImageNum += 1
        
          ##########################################################
          # Add any standardization or data augmentation here!
          ##########################################################


        I1, Label = TrainSet[RandIdx]

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(I1Batch).to(device), torch.stack(LabelBatch).to(device)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    
def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, LogsPath, TestSet, ModelNum):
    """
    Inputs: 
    ... (same as your original code)
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Initialize the model
    model = CIFAR10Model(InputSize=3*32*32, OutputSize=10, ModelNum=ModelNum) 
    model.to(device)
    # Fill your optimizer of choice here!
    Optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    Writer.add_graph(model, torch.randn(1, 3, 32, 32).to(device))

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        train_loss_this_epoch = 0
        train_acc_this_epoch = 0
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        # total_correct = 0  # Initialize total correct predictions for the epoch
        # total_samples = 0  # Initialize total processed samples for the epoch
        # loss_per_epoch = 0

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)

            # data = batch[0]
            # labels = batch[1]
            
            # outputs = model(data)
            
            # Calculate the number of correct predictions in this minibatch
            # _, predicted = torch.max(outputs, 1)
            # correct = (predicted == labels).sum().item()
            # total_correct += correct
            # total_samples += labels.size(0)

            # Predict output with forward pass
            LossThisBatch = model.training_step(batch)  # Assuming your training_step method accepts data and labels
            # loss_per_epoch += LossThisBatch

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'

                # torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
                # print('\n' + SaveName + ' Model Saved...')

            # ... (your existing code for validation_step, epoch_end, and tensorboard logging)

            result = model.validation_step(batch)
            train_loss_this_epoch += result['loss']
            train_acc_this_epoch += result['acc']
        model.epoch_end(Epochs, result)

        test_lossThisEpoch = 0
        test_AccuracyThisEpoch = 0
        model.eval()
        NumberofIterations_test = int(len(TestSet)/MiniBatchSize)
        for PerEpochCounter in tqdm(range(NumberofIterations_test)):
            Batch = GenerateBatch(TestSet, TrainLabels, ImageSize, MiniBatchSize)
            result = model.validation_step(Batch)
            test_lossThisEpoch += result["loss"]
            test_AccuracyThisEpoch += result["acc"]

        #add train and test loss and accuracy to tensorboard
        Writer.add_scalar('Train Loss', train_loss_this_epoch/NumIterationsPerEpoch, Epochs)
        Writer.add_scalar('Train Accuracy', train_acc_this_epoch/NumIterationsPerEpoch, Epochs)
        Writer.add_scalar('Test Loss', test_lossThisEpoch/NumberofIterations_test, Epochs)
        Writer.add_scalar('Test Accuracy', test_AccuracyThisEpoch/NumberofIterations_test, Epochs)
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')

        model.train()

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    SaveName = CheckPointPath + 'model.onnx'
    torch.onnx.export(model, dummy_input, SaveName, verbose=True)
    print(f"Model saved to {SaveName}")

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    # Parser.add_argument('--BasePath', default='/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/HW0/YourDirectoryID_hw0/Phase2/CIFAR10')
    Parser.add_argument('--CheckPointPath', default='Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=20, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--ModelNum', type=int, default=0, help='Model Number, Default:0')
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=ToTensor())

    TestSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=ToTensor())

    Args = Parser.parse_args()
    #BasePath = Args.BasePath
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelNum = Args.ModelNum

    #HW0/YourDirectoryID_hw0/Phase2/Checkpoints

    #CheckPointPath = "/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/HW0/YourDirectoryID_hw0/Phase2/Checkpoints"

    CheckPointPath = CheckPointPath + "Model" + str(ModelNum) + "/"
    LogsPath = LogsPath + "Model" + str(ModelNum) + "/"

    if not os.path.exists(CheckPointPath):
        os.makedirs(CheckPointPath)
    
    if not os.path.exists(LogsPath):
        os.makedirs(LogsPath)
    
    BasePath = "CIFAR10"
    
    # Setup all needed parameters including file reading
    SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath,CheckPointPath)
    #DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, LogsPath, TestSet, ModelNum)

    
if __name__ == '__main__':
    main()
 
