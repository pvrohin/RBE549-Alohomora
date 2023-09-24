#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Prof. Nitin J. Sanket (nsanket@wpi.edu)
Assistant Professor,
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True

def tic():
    """
    Function to start timer
    Tries to mimic tic() toc() in MATLAB
    """
    StartTime = time.time()
    return StartTime

def toc(StartTime):
    """
    Function to stop timer
    Tries to mimic tic() toc() in MATLAB
    """
    return time.time() - StartTime

# def FindLatestModel(CheckPointPath):
#     """
#     Finds Latest Model in CheckPointPath
#     Inputs:
#     CheckPointPath - Path where you have stored checkpoints
#     Outputs:
#     LatestFile - File Name of the latest checkpoint
#     """
#     FileList = glob.glob(CheckPointPath + '*.ckpt') # * means all if need specific format then *.csv
#     LatestFile = max(FileList, key=os.path.getctime)
#     # Strip everything else except needed information
#     LatestFile = LatestFile.replace(CheckPointPath, '')
#     LatestFile = LatestFile.replace('.ckpt', '')
#     return LatestFile


def FindLatestModel(CheckPointPath):
    """
    Finds Latest Model in CheckPointPath
    Inputs:
    CheckPointPath - Path where you have stored checkpoints
    Outputs:
    LatestFile - File Name of the latest checkpoint
    """
    FileList = glob.glob(CheckPointPath + '*.ckpt') # Search for files with the .ckpt extension
    if not FileList:
        # No checkpoint files found
        return None
    
    try:
        LatestFile = max(FileList, key=os.path.getctime)
        # Strip everything else except needed information
        LatestFile = LatestFile.replace(CheckPointPath, '')
        LatestFile = LatestFile.replace('.ckpt', '')
        return LatestFile
    except ValueError:
        # An error occurred while trying to find the latest file
        return None


def convertToOneHot(vector, NumClasses):
    """
    vector - vector of argmax indexes
    NumClasses - Number of classes
    """
    return np.equal.outer(vector, np.arange(NumClasses)).astype(np.float)


