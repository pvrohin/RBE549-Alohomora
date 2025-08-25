"""
ResNet Architecture Demonstration
This script shows step by step how the ResNet is built and how data flows through it
"""

import torch
import torch.nn as nn
import numpy as np
from Network.Network import ResnetBlock, CIFAR10Model

def print_tensor_info(name, tensor):
    """Helper function to print tensor information"""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Data type: {tensor.dtype}")
    print(f"  Min/Max values: {tensor.min().item():.4f} / {tensor.max().item():.4f}")
    print()

def demonstrate_resnet_architecture():
    """Demonstrate the ResNet architecture step by step"""
    
    print("=" * 60)
    print("RESNET ARCHITECTURE DEMONSTRATION")
    print("=" * 60)
    
    # Create a sample input (batch_size=2, channels=3, height=32, width=32)
    # This simulates a CIFAR-10 image batch
    sample_input = torch.randn(2, 3, 32, 32)
    print("SAMPLE INPUT:")
    print(f"  Shape: {sample_input.shape}")
    print(f"  Represents: 2 CIFAR-10 images (32x32 RGB)")
    print()
    
    # Initialize the ResNet model
    print("INITIALIZING RESNET MODEL...")
    model = CIFAR10Model(InputSize=32, OutputSize=10, ModelNum=2)
    print("Model initialized successfully!")
    print()
    
    # Let's examine the ResNet blocks individually
    print("=" * 60)
    print("RESNET BLOCK ARCHITECTURE")
    print("=" * 60)
    
    # Create a sample ResNet block
    print("Creating a ResNet Block (64 -> 256 channels, initial=True):")
    resnet_block = ResnetBlock(64, 256, initial=True)
    print(f"  Input channels: 64")
    print(f"  Output channels: 256")
    print(f"  Initial block: True")
    print()
    
    print("ResNet Block Structure:")
    print("  conv1: 1x1 conv (64 -> 64) - Bottleneck down")
    print("  conv2: 3x3 conv (64 -> 64) - Feature extraction")
    print("  conv3: 1x1 conv (64 -> 256) - Bottleneck up")
    print("  conv4: 1x1 conv (64 -> 256) - Identity mapping")
    print("  BatchNorm layers after each conv")
    print("  ReLU activation after each conv")
    print()
    
    # Demonstrate data flow through a single ResNet block
    print("=" * 60)
    print("DATA FLOW THROUGH RESNET BLOCK")
    print("=" * 60)
    
    # Create sample input for the block
    block_input = torch.randn(2, 64, 16, 16)
    print(f"Block Input: {block_input.shape}")
    
    # Show the forward pass step by step
    identity = block_input
    
    # Step 1: First 1x1 convolution (bottleneck down)
    x = resnet_block.conv1(block_input)
    print(f"After conv1 (1x1, 64->64): {x.shape}")
    
    # Step 2: First batch norm and ReLU
    x = resnet_block.batchnorm1(x)
    x = resnet_block.relu(x)
    print(f"After batchnorm1 + ReLU: {x.shape}")
    
    # Step 3: 3x3 convolution
    x = resnet_block.conv2(x)
    print(f"After conv2 (3x3, 64->64): {x.shape}")
    
    # Step 4: Second batch norm and ReLU
    x = resnet_block.batchnorm1(x)
    x = resnet_block.relu(x)
    print(f"After batchnorm1 + ReLU: {x.shape}")
    
    # Step 5: Third 1x1 convolution (bottleneck up)
    x = resnet_block.conv3(x)
    print(f"After conv3 (1x1, 64->256): {x.shape}")
    
    # Step 6: Final batch norm
    x = resnet_block.batchnorm2(x)
    print(f"After batchnorm2: {x.shape}")
    
    # Step 7: Identity mapping (since initial=True)
    identity = resnet_block.conv4(identity)
    identity = resnet_block.batchnorm2(identity)
    print(f"Identity branch (64->256): {identity.shape}")
    
    # Step 8: Add identity and apply ReLU
    x = x + identity
    x = resnet_block.relu(x)
    print(f"After identity addition + ReLU: {x.shape}")
    print()
    
    # Now demonstrate the full ResNet forward pass
    print("=" * 60)
    print("FULL RESNET FORWARD PASS")
    print("=" * 60)
    
    print("Step-by-step data flow through the complete ResNet:")
    
    # Initial convolution
    x = model.conv1(sample_input)
    print(f"1. After conv1 (7x7, 3->64, stride=2): {x.shape}")
    
    # Max pooling
    x = model.maxpool(x)
    print(f"2. After maxpool (3x3, stride=2): {x.shape}")
    
    # First ResNet block (initial)
    x = model.resnetblock1_d(x)
    print(f"3. After resnetblock1_d (64->256, initial): {x.shape}")
    
    # Two more ResNet blocks
    for i in range(2):
        x = model.resnetblock1(x)
        print(f"4.{i+1}. After resnetblock1 (256->256): {x.shape}")
    
    # Second stage (downsample)
    x = model.resnetblock2_d(x)
    print(f"5. After resnetblock2_d (256->512, downsample): {x.shape}")
    
    # Three more blocks in second stage
    for i in range(3):
        x = model.resnetblock2(x)
        print(f"6.{i+1}. After resnetblock2 (512->512): {x.shape}")
    
    # Third stage (downsample)
    x = model.resnetblock3_d(x)
    print(f"7. After resnetblock3_d (512->1024, downsample): {x.shape}")
    
    # Five more blocks in third stage
    for i in range(5):
        x = model.resnetblock3(x)
        print(f"8.{i+1}. After resnetblock3 (1024->1024): {x.shape}")
    
    # Fourth stage (downsample)
    x = model.resnetblock4_d(x)
    print(f"9. After resnetblock4_d (1024->2048, downsample): {x.shape}")
    
    # Two more blocks in fourth stage
    for i in range(2):
        x = model.resnetblock4(x)
        print(f"10.{i+1}. After resnetblock4 (2048->2048): {x.shape}")
    
    # Global average pooling
    x = model.avgpool(x)
    print(f"11. After adaptive avgpool: {x.shape}")
    
    # Flatten
    x = x.view(-1, 2048)
    print(f"12. After flatten: {x.shape}")
    
    # Final classification layer
    x = model.fc1(x)
    print(f"13. After fc1 (2048->10): {x.shape}")
    
    print()
    print("=" * 60)
    print("ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    print("ResNet-50 Architecture:")
    print("  - Input: 3x32x32 (CIFAR-10 images)")
    print("  - Initial conv: 7x7, 3->64, stride=2")
    print("  - MaxPool: 3x3, stride=2")
    print("  - Stage 1: 3 blocks (64->256)")
    print("  - Stage 2: 4 blocks (256->512)")
    print("  - Stage 3: 6 blocks (512->1024)")
    print("  - Stage 4: 3 blocks (1024->2048)")
    print("  - Global Average Pooling")
    print("  - Fully Connected: 2048->10")
    print()
    
    print("Key Features:")
    print("  - Bottleneck design: 1x1 -> 3x3 -> 1x1 convolutions")
    print("  - Identity connections (skip connections)")
    print("  - Downsampling at stage transitions")
    print("  - Batch normalization after each convolution")
    print("  - ReLU activation functions")
    
    print()
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_resnet_architecture()
