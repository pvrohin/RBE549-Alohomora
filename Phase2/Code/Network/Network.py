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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out,labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels,initial=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial = initial
        if in_channels != out_channels and not initial:
            self.conv1= nn.Conv2d(in_channels, int(out_channels/4), 1,2)
            self.conv2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), 3,1,1)
            self.conv3 = nn.Conv2d(int(out_channels/4), out_channels, 1,1)
            self.conv4 = nn.Conv2d(in_channels, out_channels,1,2)
            self.batchnorm1 = nn.BatchNorm2d(int(out_channels/4))
            self.batchnorm2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        elif initial:
            self.conv1= nn.Conv2d(in_channels, int(out_channels/4), 1)
            self.conv2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), 3,1,1)
            self.conv3 = nn.Conv2d(int(out_channels/4), out_channels, 1,1)
            self.conv4 = nn.Conv2d(in_channels, out_channels,1)
            self.batchnorm1 = nn.BatchNorm2d(int(out_channels/4))
            self.batchnorm2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        else:
            self.conv1= nn.Conv2d(in_channels, int(out_channels/4), 1)
            self.conv2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), 3,1,1)
            self.conv3 = nn.Conv2d(int(out_channels/4), out_channels, 1)
            self.batchnorm1 = nn.BatchNorm2d(int(out_channels/4))
            self.batchnorm2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        
    def forward(self, xb):
        identity = xb
        xb = self.conv1(xb)
        xb = self.batchnorm1(xb)
        xb = self.relu(xb)
        xb = self.conv2(xb)
        xb = self.batchnorm1(xb)
        xb = self.relu(xb)
        xb = self.conv3(xb)
        xb = self.batchnorm2(xb)


        
        if self.in_channels != self.out_channels or self.initial:
            identity = self.conv4(identity)
            identity = self.batchnorm2(identity)

        xb += identity
        xb = self.relu(xb)
        return xb

class ResnextBlock(nn.Module):
    def __init__(self, in_channels, out_channels,initial=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial = initial
        self.cardinality = 8
        self.bottleneck= 14
        self.intermeditate_channels= self.bottleneck * self.cardinality
        if in_channels != out_channels and not initial:
            self.conv1= nn.Conv2d(in_channels, self.bottleneck, 1,2)
            self.conv2 = nn.Conv2d(self.bottleneck, self.bottleneck, 3,1,1)
            self.conv3 = nn.Conv2d(self.intermeditate_channels, out_channels, 1,1)
            self.conv4 = nn.Conv2d(in_channels, out_channels,1,2)
            self.batchnorm1 = nn.BatchNorm2d(self.bottleneck)
            self.batchnorm2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        elif initial:
            self.conv1= nn.Conv2d(in_channels, self.bottleneck, 1)
            self.conv2 = nn.Conv2d(self.bottleneck, self.bottleneck, 3,1,1)
            self.conv3 = nn.Conv2d(self.intermeditate_channels, out_channels, 1,1)
            self.conv4 = nn.Conv2d(in_channels, out_channels,1)
            self.batchnorm1 = nn.BatchNorm2d(self.bottleneck)
            self.batchnorm2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        else:
            self.conv1= nn.Conv2d(in_channels, self.bottleneck, 1)
            self.conv2 = nn.Conv2d(self.bottleneck, self.bottleneck, 3,1,1)
            self.conv3 = nn.Conv2d(self.intermeditate_channels, out_channels, 1)
            self.batchnorm1 = nn.BatchNorm2d(self.bottleneck)
            self.batchnorm2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        
    def forward(self, xb):
      identity = xb
      out=[]
      inp=xb
      for i in range(self.cardinality):
        xb = self.conv1(inp)
        xb = self.batchnorm1(xb)
        xb = self.relu(xb)
        xb = self.conv2(xb)
        xb = self.batchnorm1(xb)
        xb = self.relu(xb)
        out.append(xb)
        
        out = torch.cat(out, dim=1)
        out = self.conv3(out)
        out = self.batchnorm2(out)

        
        if self.in_channels != self.out_channels or self.initial:
            identity = self.conv4(identity)
            identity = self.batchnorm2(identity)
        
        out += identity
        out = self.relu(out)
        return out



class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize, ModelNum):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      #############################
      # Fill your network initialization of choice here!
      #############################
      super().__init__()
      if ModelNum == 0:
        self.init_basenet()
      elif ModelNum == 1:
        self.init_improvednet()
      elif ModelNum == 2:
        self.init_resnet()
      elif ModelNum == 3:
        self.init_resnext()
      # elif ModelNum == 4:
      #   self.init_densenet()


  def init_basenet(self):
    self.conv1 = nn.Conv2d(3, 32, 3,1,1)
    self.conv2 = nn.Conv2d(32, 64, 3,1,1)
    self.maxpool= nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64*8*8, 128)
    self.fc2 = nn.Linear(128, 10)
        
  def basenet_forward(self, xb):
     xb = self.conv1(xb)
     xb = F.relu(xb)
     xb = self.maxpool(xb)
     xb = self.conv2(xb)
     xb = F.relu(xb)
     xb = self.maxpool(xb)
     xb = xb.view(-1, 64*8*8)
     xb = self.fc1(xb)
     xb = self.fc2(xb)
     return xb

  def init_improvednet(self):
    self.conv1 = nn.Conv2d(3, 32, 3,1,1)
    self.conv2 = nn.Conv2d(32, 64, 3,1,1)
    self.conv3 = nn.Conv2d(64, 128, 3,1,1)
    self.maxpool= nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(128*4*4, 128)
    self.fc2 = nn.Linear(128, 10)
    self.batchnorm1 = nn.BatchNorm2d(32)
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.batchnorm3 = nn.BatchNorm2d(128)

  def improvednet_forward(self, xb):
    xb = self.conv1(xb)
    xb = self.batchnorm1(xb)
    xb = F.relu(xb)
    xb = self.maxpool(xb)
    xb = self.conv2(xb)
    xb = self.batchnorm2(xb)
    xb = F.relu(xb)
    xb = self.maxpool(xb)
    xb = self.conv3(xb)
    xb = self.batchnorm3(xb)
    xb = F.relu(xb)
    xb = self.maxpool(xb)
    xb = xb.view(-1, 128*4*4)
    xb = self.fc1(xb)
    xb = self.fc2(xb)
    return xb

  def forward(self, xb):
    return self.improvednet_forward(xb)

  def init_resnet(self):
    #initial a blocks and layers needed for resnet 50
    self.conv1 = nn.Conv2d(3, 64, 7,2)
    self.maxpool= nn.MaxPool2d(3, 2)
    self.resnetblock1_d = ResnetBlock(64, 256,initial=True)
    self.resnetblock1 = ResnetBlock(256, 256)
    self.resnetblock2_d = ResnetBlock(256, 512)
    self.resnetblock2 = ResnetBlock(512, 512)
    self.resnetblock3_d = ResnetBlock(512, 1024)
    self.resnetblock3 = ResnetBlock(1024, 1024)
    self.resnetblock4_d = ResnetBlock(1024, 2048)
    self.resnetblock4 = ResnetBlock(2048, 2048)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Linear(2048, 10)

  def resnet_forward(self, xb):
    xb = self.conv1(xb)
    xb = self.maxpool(xb)
    xb = self.resnetblock1_d(xb)
    for i in range(2):
      xb = self.resnetblock1(xb)
    xb = self.resnetblock2_d(xb)
    for i in range(3):
      xb = self.resnetblock2(xb)
    xb = self.resnetblock3_d(xb)
    for i in range(5):
      xb = self.resnetblock3(xb)
    xb = self.resnetblock4_d(xb)
    for i in range(2):
      xb = self.resnetblock4(xb)
    xb = self.avgpool(xb)
    xb = xb.view(-1, 2048)
    xb = self.fc1(xb)
    return xb


  def init_resnext(self):
    #initial a blocks and layers needed for resnet 50
    self.conv1 = nn.Conv2d(3, 64, 7,2)
    self.maxpool= nn.MaxPool2d(3, 2)
    self.resnetblock1_d = ResnextBlock(64, 256,initial=True)
    self.resnetblock1 = ResnextBlock(256, 256)
    self.resnetblock2_d = ResnextBlock(256, 512)
    self.resnetblock2 = ResnextBlock(512, 512)
    self.resnetblock3_d = ResnextBlock(512, 1024)
    self.resnetblock3 = ResnextBlock(1024, 1024)
    self.resnetblock4_d = ResnextBlock(1024, 2048)
    self.resnetblock4 = ResnextBlock(2048, 2048)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Linear(2048, 10)
    
  def resnext_forward(self, xb):
    xb = self.conv1(xb)
    xb = self.maxpool(xb)
    xb = self.resnetblock1_d(xb)
    for i in range(2):
      xb = self.resnetblock1(xb)
    xb = self.resnetblock2_d(xb)
    for i in range(3):
      xb = self.resnetblock2(xb)
        
    xb = self.resnetblock3_d(xb)
    for i in range(5):
      xb = self.resnetblock3(xb)

    xb = self.resnetblock4_d(xb)
    for i in range(2):
      xb = self.resnetblock4(xb)

    xb = self.avgpool(xb)
    xb = xb.view(-1, 2048)
    xb = self.fc1(xb)
    return xb

  def forward(self, xb):
    """
    Input:
    xb is a MiniBatch of the current image
    Outputs:
    out - output of the network
    """
    #############################
    # Fill your network structure of choice here!
    #############################
    if self.ModelNum == 0:
      out = self.basenet_forward(xb)
    elif self.ModelNum == 1:
      out = self.improvednet_forward(xb)
    elif self.ModelNum == 2:
      out = self.resnet_forward(xb)   
    elif self.ModelNum == 3:
      out = self.resnext_forward(xb)   
    # elif self.ModelNum == 4:
    #   out = self.densenet_forward(xb)
        
    return out
       

