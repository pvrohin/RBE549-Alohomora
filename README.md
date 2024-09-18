# RBE549-Alohomora
A simplified version of pb edge detection, which finds boundaries by examining brightness, color, and texture information across multiple scales (different sizes of objects/image)

# Probability-Based Edge Detection

This repository contains an implementation of a simple version of the Probability-Based (PB) boundary detection algorithm. Unlike classical approaches like Canny and Sobel, which only measure image intensity discontinuities, the PB algorithm also considers texture and color information, resulting in improved performance in edge detection.

## Sample Input Image
![Sample Input](https://github.com/pvrohin/RBE549-Alohomora/Phase1/BSDS500/Images/1.jpg)

## Filter Banks
Filter banks are a set of filters applied over an image to extract multiple features. In this project, filter banks were used to extract texture properties. Below are the implementations of DoG filters, Leung-Malik filters, and Gabor filters.

### Oriented Derivative of Gaussian (DoG) Filters
![DoG Filters](https://github.com/pvrohin/RBE549-Alohomora/Phase1/Output/DoG.png)

### Leung-Malik Filters (LM Large and LM Small)
![LM Filters](https://github.com/pvrohin/RBE549-Alohomora/Phase1/Output/LM.png)

### Gabor Filters
![Gabor Filters](https://github.com/pvrohin/RBE549-Alohomora/Phase1/Output/Gabor.png)

## Texton Maps
The filters are used to detect texture properties in an image. By clustering filter responses using the K-means algorithm (K=64), similar texture properties are grouped into texton maps.

![Texton Map](https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Textron_map/TextonMap_10.jpg)

## Brightness Maps
Each pixel's brightness is clustered using the K-means algorithm (K=16) to generate brightness maps.

![Brightness Map](https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Brightness_map/BrightnessMap_10.jpg)

## Color Maps
The image is clustered based on RGB values using K-means (K=16) to create color maps.

![Color Map](https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Color_map/ColorMap_10.jpg)

## Texture, Brightness, and Color Gradients
To compute gradients of texture, brightness, and color (Tg, Bg, Cg), Half-disc masks were used to capture differences across shapes and sizes.

### Half-Disc Masks
![Half-Disc Masks](https://github.com/pvrohin/RBE549-Alohomora/Phase1/Output/HD.png)

### Texture Gradient
![Texture Gradient](https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/T_g/tg_10.jpg)

### Brightness Gradient
![Brightness Gradient](https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/B_g/bg_10.jpg)

### Color Gradient
![Color Gradient](https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/C_g/cg_10.jpg)

## Sobel and Canny Baseline
The Sobel and Canny edge detectors were combined using a weighted average method to serve as the baseline.

### Sobel Baseline
![Sobel Baseline](https://github.com/pvrohin/RBE549-Alohomora/Phase1/BSDS500/SobelBaseline/1.png)

### Canny Baseline
![Canny Baseline](https://github.com/pvrohin/RBE549-Alohomora/Phase1/BSDS500/CannyBaseline/1.png)

## Pb-lite Output
In the final step, the features from the baseline methods (Canny and Sobel) were combined with the gradients of texture (τ), brightness (β), and color (ζ) to generate the PB-lite output.

![Pb-lite Output](https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/pb_lite_output/10.png)

## Running the Code
### File Structure
