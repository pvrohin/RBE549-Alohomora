#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import matplotlib.pyplot as plt
import imutils
import math
import os
import glob
import cv2

def Gaussian_1D(scale=1,size=7):
	sigma = scale

	#Generate x and y coordinate lists as centered in the grid of given size
	x = np.arange(-size/2+0.5,size/2)
	y = np.arange(-size/2+0.5,size/2)

	#Create a grid using the above defined axis in order to center the gaussian
	X,Y = np.meshgrid(x,y)

	#Calculate the Gaussian
	Gaussian = np.exp(-(X**2+Y**2)/(2*sigma**2))

	#Return the Normalized Gaussian
	return Gaussian/np.sum(Gaussian)

def Gaussian_2D(mu1=0,mu2=0,size=7,covariance=[1,1]):
	sigma_1 = covariance[0]
	sigma_2 = covariance[1] 

	#Generate x and y coordinate lists as centered in the grid of given size
	x = np.arange(-size/2+0.5,size/2)
	y = np.arange(-size/2+0.5,size/2)

	#Create a grid using the above defined axis in order to center the gaussian
	X,Y = np.meshgrid(x,y)

	#Calculate the Gaussian
	#Gaussian = #Gaussian_1D(sigma_1,size)*Gaussian_1D(sigma_2,size)#
	Gaussian = np.exp(-((X-mu1)**2)/(2*sigma_1**2))*np.exp(-((Y-mu2)**2)/(2*sigma_2**2))
	
	# plt.imshow(Gaussian)
	# plt.show()

	#Return the Normalized Gaussian
	return Gaussian/np.sum(Gaussian)

def gabor_kernel(size=37, Lambda=5*np.pi, theta=0, sigma=20, gamma=0.7):
	#Generate x and y coordinate lists as centered in the grid of given size
	x = np.arange(-size/2+0.5,size/2)
	y = np.arange(-size/2+0.5,size/2)

	#Create a grid using the above defined axis in order to center the gaussian
	X,Y = np.meshgrid(x,y)

	X_hat = X*np.cos(theta)+Y*np.sin(theta)

	Y_hat = -X*np.sin(theta)+Y*np.cos(theta)

	Gabor = np.exp(-((X_hat**2)+(gamma**2)*(Y_hat**2))/(2*(sigma**2)))*np.cos(((2*np.pi*X_hat)/(Lambda))+sigma)

	return Gabor

def gabor_FilterBank(size=37):

	angles = []
	orientations = 8
	Lambdas = [2,4,6,8,10]
	Sigmas = [4,8,10,12,14]

	Gabor_filter_bank = np.zeros((size,size,len(Sigmas)*orientations))

	no_of_filters = 0

	#Generate angles for rotation according to number of orientations
	for i in range(orientations):
		angles.append(i*(360/orientations))

	#print(angles)

	for i in range(len(Sigmas)):
		#Rotate the generated Filter according to number of orientations
		for angle in angles:
			Gabor_filter_bank[:,:,no_of_filters] = gabor_kernel(size,Lambdas[i],angle,Sigmas[i])
			no_of_filters = no_of_filters+1

	return Gabor_filter_bank

def half_disk_mask(sizes = [10,20,30],orientations = 8):

	Half_Disk_bank = []#np.zeros((size,size,len(sizes)*orientations))

	no_of_masks = 0

	angles = []
	#Generate angles for rotation according to number of orientations
	for i in range(orientations):
		angles.append(i*(360/orientations))

	for size in sizes:
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

		for angle in angles:
			Half_Disk_bank.append(imutils.rotate(mask,angle))
			Half_Disk_bank.append(imutils.rotate(mask,angle-180))
			no_of_masks = no_of_masks + 2

	# plt.imshow(mask, cmap='gray')
	# plt.show()
	return Half_Disk_bank


def LM_FilterBank(size=37,total_no_of_filters=48):
	no_of_filters = 0

	LM_filter_bank = np.zeros((size,size,total_no_of_filters))

	Sobel_Filter = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float32)

	##########################
	covariance = [3,9]

	Gaussian = Gaussian_2D(size=size,covariance=covariance)

	print(np.shape(Gaussian))

	#Scale 2 degree 1
	result = cv2.filter2D(Gaussian,-1,Sobel_Filter)

	# plt.imshow(result,cmap="gray")
	# plt.show()

	angles = []

	orientations = 6

	for i in range(orientations):
		angles.append(i*(360/orientations))

	for angle in angles:
			LM_filter_bank[:,:,no_of_filters] = imutils.rotate(result,angle)
			no_of_filters = no_of_filters+1

	result = Gaussian

	#Scale 2 degree 2
	for degree in range(2):
		result = cv2.filter2D(result,-1,Sobel_Filter)

	angles = []

	orientations = 6

	for i in range(orientations):
		angles.append(i*(360/orientations))

	for angle in angles:
			LM_filter_bank[:,:,no_of_filters] = imutils.rotate(result,angle)
			no_of_filters = no_of_filters+1

    ##########################
	covariance = [np.sqrt(2),3*np.sqrt(2)]
	Gaussian = Gaussian_2D(size=size,covariance=covariance)

	#Scale sqrt2 degree 1
	result = cv2.filter2D(Gaussian,-1,Sobel_Filter)

	# plt.imshow(result,cmap="gray")
	# plt.show()

	angles = []

	orientations = 6

	for i in range(orientations):
		angles.append(i*(360/orientations))

	for angle in angles:
			LM_filter_bank[:,:,no_of_filters] = imutils.rotate(result,angle)
			no_of_filters = no_of_filters+1

	result = Gaussian

	#Scale sqrt2 degree 2
	for degree in range(2):
		result = cv2.filter2D(result,-1,Sobel_Filter)

	angles = []

	orientations = 6

	for i in range(orientations):
		angles.append(i*(360/orientations))

	for angle in angles:
			LM_filter_bank[:,:,no_of_filters] = imutils.rotate(result,angle)
			no_of_filters = no_of_filters+1

	##########################
	covariance = [1,3]
	Gaussian = Gaussian_2D(size=size,covariance=covariance)

	#Scale sqrt2 degree 1
	result = cv2.filter2D(Gaussian,-1,Sobel_Filter)

	# plt.imshow(result,cmap="gray")
	# plt.show()

	angles = []

	orientations = 6

	for i in range(orientations):
		angles.append(i*(360/orientations))

	for angle in angles:
			LM_filter_bank[:,:,no_of_filters] = imutils.rotate(result,angle)
			no_of_filters = no_of_filters+1

	result = Gaussian

	#Scale sqrt2 degree 2
	for degree in range(2):
		result = cv2.filter2D(result,-1,Sobel_Filter)

	angles = []

	orientations = 6

	for i in range(orientations):
		angles.append(i*(360/orientations))

	for angle in angles:
			LM_filter_bank[:,:,no_of_filters] = imutils.rotate(result,angle)
			no_of_filters = no_of_filters+1

	##################
	LaPlacian_Filter = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=np.float32)

	Gaussian = Gaussian_1D(1,size=size)
	result = cv2.filter2D(Gaussian,-1,LaPlacian_Filter)
	LM_filter_bank[:,:,no_of_filters] = result
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(np.sqrt(2),size=size)
	result = cv2.filter2D(Gaussian,-1,LaPlacian_Filter)
	LM_filter_bank[:,:,no_of_filters] = result
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(2,size=size)
	result = cv2.filter2D(Gaussian,-1,LaPlacian_Filter)
	LM_filter_bank[:,:,no_of_filters] = result
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(2*np.sqrt(2),size=size)
	result = cv2.filter2D(Gaussian,-1,LaPlacian_Filter)
	LM_filter_bank[:,:,no_of_filters] = result
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(3,size=size)
	result = cv2.filter2D(Gaussian,-1,LaPlacian_Filter)
	LM_filter_bank[:,:,no_of_filters] = result
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(3*np.sqrt(2),size=size)
	result = cv2.filter2D(Gaussian,-1,LaPlacian_Filter)
	LM_filter_bank[:,:,no_of_filters] = result
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(6,size=size)
	result = cv2.filter2D(Gaussian,-1,LaPlacian_Filter)
	LM_filter_bank[:,:,no_of_filters] = result
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(6*np.sqrt(2),size=size)
	result = cv2.filter2D(Gaussian,-1,LaPlacian_Filter)
	LM_filter_bank[:,:,no_of_filters] = result
	no_of_filters = no_of_filters+1

	#########################
	Gaussian = Gaussian_1D(1,size=size)
	LM_filter_bank[:,:,no_of_filters] = Gaussian
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(np.sqrt(2),size=size)
	LM_filter_bank[:,:,no_of_filters] = Gaussian
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(2,size=size)
	LM_filter_bank[:,:,no_of_filters] = Gaussian
	no_of_filters = no_of_filters+1

	Gaussian = Gaussian_1D(2*np.sqrt(2),size=size)
	LM_filter_bank[:,:,no_of_filters] = Gaussian
	no_of_filters = no_of_filters+1

	print(no_of_filters)

	return LM_filter_bank

def DoG(size=7,scales=[1,2],orientations=16):

	#Declare Sobel Filter
	Sobel_Filter = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float32)

	angles = []

	DoG_filter_bank = np.zeros((size,size,len(scales)*orientations))

	no_of_filters = 0

	#Generate angles for rotation according to number of orientations
	for i in range(orientations):
		angles.append(i*(360/orientations))

	print(angles)

	for scale in scales:

		#Generate Gaussian
		Gaussian = Gaussian_1D(scale,size)
		
		#Convolve Gaussian and Sobel Filter
		dog = cv2.filter2D(Gaussian,-1,Sobel_Filter)

		#Rotate the generated Filter according to number of orientations
		for angle in angles:
			DoG_filter_bank[:,:,no_of_filters] = imutils.rotate(dog,angle)
			no_of_filters = no_of_filters+1

	return DoG_filter_bank

def TextonMap(image,DoGFilterBank,LMFilterBank,GaborFilterBank):
	dog_filters =  DoGFilterBank.shape[2]
	lm_filters = LMFilterBank.shape[2]
	gabor_filters = GaborFilterBank.shape[2]

	total_no_of_filters = dog_filters + lm_filters + gabor_filters

	filter_count = 0

	final_filtered_response = np.zeros((image.shape[0],image.shape[1],total_no_of_filters))

	for i in range(dog_filters):
		final_filtered_response[:,:,filter_count] = cv2.filter2D(image,-1,DoGFilterBank[:,:,i])
		filter_count = filter_count + 1

	for i in range(lm_filters):
		final_filtered_response[:,:,filter_count] = cv2.filter2D(image,-1,LMFilterBank[:,:,i])
		filter_count = filter_count + 1

	for i in range(gabor_filters):
		final_filtered_response[:,:,filter_count] = cv2.filter2D(image,-1,GaborFilterBank[:,:,i])
		filter_count = filter_count + 1

	flattenedResponse = final_filtered_response.reshape((-1,total_no_of_filters)).astype(np.float32)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 64
	ret,label,center=cv2.kmeans(flattenedResponse,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	label = label.reshape(image.shape[0], image.shape[1])

	return label

	# plt.imshow(label)
	# plt.show()

def BrightnessMap(image):
	flattenedResponse = image.reshape((-1,1)).astype(np.float32)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(flattenedResponse,16,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
	label = label.reshape(image.shape[0],image.shape[1])
    
	# plt.imshow(label)
	# plt.show()
	
	return label

def ColorMap(image):
	flattenedResponse = image.reshape((-1,3)).astype(np.float32)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(flattenedResponse,16,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
	label = label.reshape(image.shape[0],image.shape[1])
    
	# plt.imshow(label)
	# plt.show()
	
	return label

def calc_gradient(image,halfDiskBank):
	total_no_of_disks = len(halfDiskBank)

	gradient = np.zeros((image.shape[0], image.shape[1],total_no_of_disks//2), dtype=np.float32)

	count = 0

	epsilon = np.ones((image.shape),dtype=np.float32)*1e-7

	for i in range(0,total_no_of_disks,2):
		chi_sqr_dist = np.zeros((image.shape[0],image.shape[1]),dtype=np.float32)

		left_mask, right_mask = halfDiskBank[i], halfDiskBank[i+1]

		for index in range(64):
			tmp = image.copy()
			tmp[image == index] = 1.0
			tmp[image != index] = 0.0			
			g_i = cv2.filter2D(tmp, -1, left_mask)
			h_i = cv2.filter2D(tmp, -1, right_mask)

			chi_sqr_dist = chi_sqr_dist + (((g_i-h_i)**2)/(g_i+h_i+epsilon))

		gradient[:,:,count] = 0.5*chi_sqr_dist
		count = count + 1

	gradient_mean = gradient.mean(axis=2, dtype=np.float32)

	# plt.imshow(tgmean)
	# plt.show()
	
	return gradient_mean			

def main():
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	size = 15
	scales = [1,2]
	orientations = 16
	DoG_filter_bank = DoG(size,scales,orientations)

	# Save DoG filter bank
	fig1 = plt.figure(figsize=(20, 4))
	for i in range(1, len(scales)*orientations+1):
		ax = fig1.add_subplot(2, 16, i)			
		plt.imshow(DoG_filter_bank[:,:,i-1], interpolation='none', cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])

	fig1.suptitle("DoG Filter Bank", fontsize=20)
	plt.savefig('../Output/DoG.png', dpi=150, bbox_inches='tight')
	plt.close()

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

	lm_FilterBank = LM_FilterBank()

	# Save LM filter bank
	fig1 = plt.figure(figsize=(20, 8))
	for i in range(1, 49):
		ax = fig1.add_subplot(4, 12, i)			
		plt.imshow(lm_FilterBank[:,:,i-1], interpolation='none', cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])

	fig1.suptitle("LM Filter Bank", fontsize=20)
	plt.savefig('../Output/LM.png', dpi=150, bbox_inches='tight')
	plt.close()

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	Gabor_FilterBank = gabor_FilterBank()

	# Save Gabor filter bank
	fig1 = plt.figure(figsize=(20, 12))
	for i in range(1, 41):
		ax = fig1.add_subplot(5, 8, i)			
		plt.imshow(Gabor_FilterBank[:,:,i-1], interpolation='none', cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])

	fig1.suptitle("Gabor Filter Bank", fontsize=20)
	plt.savefig('../Output/Gabor.png', dpi=150, bbox_inches='tight')
	plt.close()

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""

	Half_disk_mask = half_disk_mask()
	
	# Save half-disk masks
	fig1 = plt.figure(figsize=(20, 15))
	for i in range(1, len(Half_disk_mask)+1):
		ax = fig1.add_subplot(6, 8, i)			
		plt.imshow(Half_disk_mask[i-1], interpolation='none', cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])

	fig1.suptitle("Half Disk Bank", fontsize=20)
	plt.savefig('../Output/HD.png', dpi=150, bbox_inches='tight')
	plt.close()

	# Get all image files from the Images folder
	image_folder = '../BSDS500/Images/'
	image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
	
	print(f"Found {len(image_files)} images to process:")
	for img_file in image_files:
		print(f"  - {os.path.basename(img_file)}")

	# Process each image
	for img_file in image_files:
		# Extract image number from filename (e.g., "7.jpg" -> "7")
		img_name = os.path.splitext(os.path.basename(img_file))[0]
		
		print(f"\nProcessing image {img_name}...")
		
		# Create output folder for this image
		output_folder = f'../Output/{img_name}/'
		os.makedirs(output_folder, exist_ok=True)
		
		# Load image
		image = cv2.imread(img_file)
		if image is None:
			print(f"Warning: Could not load image {img_file}")
			continue
			
		imageGray = image[:,:,0]	
		
		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		texton_map = TextonMap(imageGray, DoG_filter_bank, lm_FilterBank, Gabor_FilterBank)	

		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		# Save texton map
		plt.figure(figsize=(10, 8))
		plt.imshow(texton_map, cmap='tab20')
		plt.title(f'Texton Map - Image {img_name}')
		plt.axis('off')
		plt.savefig(f'{output_folder}TextonMap_{img_name}.png', dpi=150, bbox_inches='tight')
		plt.close()

		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		tg = calc_gradient(texton_map, Half_disk_mask)
		
		# Save texton gradient
		plt.figure(figsize=(10, 8))
		plt.imshow(tg, cmap='gray')
		plt.title(f'Texton Gradient (Tg) - Image {img_name}')
		plt.axis('off')
		plt.savefig(f'{output_folder}Tg_{img_name}.png', dpi=150, bbox_inches='tight')
		plt.close()

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		brightness_map = BrightnessMap(imageGray)
		
		# Save brightness map
		plt.figure(figsize=(10, 8))
		plt.imshow(brightness_map, cmap='tab20')
		plt.title(f'Brightness Map - Image {img_name}')
		plt.axis('off')
		plt.savefig(f'{output_folder}BrightnessMap_{img_name}.png', dpi=150, bbox_inches='tight')
		plt.close()

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		bg = calc_gradient(brightness_map, Half_disk_mask)
		
		# Save brightness gradient
		plt.figure(figsize=(10, 8))
		plt.imshow(bg, cmap='gray')
		plt.title(f'Brightness Gradient (Bg) - Image {img_name}')
		plt.axis('off')
		plt.savefig(f'{output_folder}Bg_{img_name}.png', dpi=150, bbox_inches='tight')
		plt.close()

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		color_map = ColorMap(image)
		
		# Save color map
		plt.figure(figsize=(10, 8))
		plt.imshow(color_map, cmap='tab20')
		plt.title(f'Color Map - Image {img_name}')
		plt.axis('off')
		plt.savefig(f'{output_folder}ColorMap_{img_name}.png', dpi=150, bbox_inches='tight')
		plt.close()

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		cg = calc_gradient(color_map, Half_disk_mask)
		
		# Save color gradient
		plt.figure(figsize=(10, 8))
		plt.imshow(cg, cmap='gray')
		plt.title(f'Color Gradient (Cg) - Image {img_name}')
		plt.axis('off')
		plt.savefig(f'{output_folder}Cg_{img_name}.png', dpi=150, bbox_inches='tight')
		plt.close()

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobel_path = f'../BSDS500/SobelBaseline/{img_name}.png'
		sobel_baseline = cv2.imread(sobel_path, 0)
		
		if sobel_baseline is not None:
			# Save Sobel baseline
			plt.figure(figsize=(10, 8))
			plt.imshow(sobel_baseline, cmap='gray')
			plt.title(f'Sobel Baseline - Image {img_name}')
			plt.axis('off')
			plt.savefig(f'{output_folder}SobelBaseline_{img_name}.png', dpi=150, bbox_inches='tight')
			plt.close()
		else:
			print(f"Warning: Could not load Sobel baseline for image {img_name}")

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		canny_path = f'../BSDS500/CannyBaseline/{img_name}.png'
		canny_baseline = cv2.imread(canny_path, 0)
		
		if canny_baseline is not None:
			# Save Canny baseline
			plt.figure(figsize=(10, 8))
			plt.imshow(canny_baseline, cmap='gray')
			plt.title(f'Canny Baseline - Image {img_name}')
			plt.axis('off')
			plt.savefig(f'{output_folder}CannyBaseline_{img_name}.png', dpi=150, bbox_inches='tight')
			plt.close()
		else:
			print(f"Warning: Could not load Canny baseline for image {img_name}")

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		if sobel_baseline is not None and canny_baseline is not None:
			pb = np.multiply(((tg+bg+cg)/3), ((0.5*sobel_baseline)+(0.5*canny_baseline)))
			
			# Save final Pb-Lite output
			plt.figure(figsize=(10, 8))
			plt.imshow(pb, cmap='gray')
			plt.title(f'Pb-Lite Edge Detection Result - Image {img_name}')
			plt.axis('off')
			plt.savefig(f'{output_folder}PbLite_{img_name}.png', dpi=150, bbox_inches='tight')
			plt.close()
		else:
			print(f"Warning: Skipping Pb-Lite computation for image {img_name} due to missing baseline files")
		
		# Also save the original image for reference
		plt.figure(figsize=(10, 8))
		plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		plt.title(f'Original Image - {img_name}')
		plt.axis('off')
		plt.savefig(f'{output_folder}Original_{img_name}.png', dpi=150, bbox_inches='tight')
		plt.close()
		
		# Save grayscale version
		plt.figure(figsize=(10, 8))
		plt.imshow(imageGray, cmap='gray')
		plt.title(f'Grayscale Image - {img_name}')
		plt.axis('off')
		plt.savefig(f'{output_folder}Grayscale_{img_name}.png', dpi=150, bbox_inches='tight')
		plt.close()
		
		print(f"Completed processing image {img_name}. Outputs saved to {output_folder}")
	
	print("\n" + "="*60)
	print("PROCESSING COMPLETE!")
	print("="*60)
	print("All outputs have been saved:")
	print("- Filter banks: DoG.png, LM.png, Gabor.png, HD.png (in main Output folder)")
	print("- Individual image results: Each image has its own numbered folder (1/, 2/, ..., 10/)")
	print("  containing all maps, gradients, baselines, and final Pb-Lite results.")
	print("="*60)


if __name__ == '__main__':
    main()
 


