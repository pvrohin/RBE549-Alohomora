import numpy as np
import matplotlib.pyplot as plt

size = 27
radius = size//2  
center = (size//2, size//2) 

x = np.arange(size)
y = np.arange(size)

# Create a mesh grid of coordinates
X, Y = np.meshgrid(x, y)

# Calculate distance from the center of the half disk to each point on the grid
R = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

# Create the half disk mask
mask = np.zeros((size,size))
mask[(R <= radius)]=1 
mask[:size//2+1,:] = 0

plt.imshow(mask, cmap='gray')
plt.show()
