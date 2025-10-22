# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the required libraries.

### Step2
Convert the image from BGR to RGB.

### Step3
Apply the required filters for the image separately.

### Step4
Plot the original and filtered image by using matplotlib.pyplot.

### Step5
End the program.

## Program:

### Name: Nithilan S
### Register Number: 212223240108



#### Convolution Result
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread('cat39.jpeg')

kernel = np.ones((5,5), dtype = np.float32) / 5**2
print (kernel)

image = cv2.imread('cat39.jpeg')

dst = cv2.filter2D(image, ddepth = -1, kernel = kernel)

plt.figure(figsize = [20,10])
plt.subplot(121); plt.axis('off'); plt.imshow(image[:,:,::-1]); plt.title("Original Image")
plt.subplot(122); plt.axis('off'); plt.imshow(dst[:,:,::-1]);   plt.title("Convolution Result")


```
#### Output
<img width="877" height="416" alt="image" src="https://github.com/user-attachments/assets/ac066687-cb9a-4161-b757-472211254ee1" />

### 1. Smoothing Filters

#### i) Using Averaging Filter

```python
average_filter = cv2.blur(image, (30,30))

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(average_filter[:, :, ::-1]); plt.title('Output Image ( Average Filter)')
```

#### Output
<img width="879" height="359" alt="image" src="https://github.com/user-attachments/assets/7cb8211c-03c4-4db0-966c-9cbf99a0fa42" />


#### ii) Using Weighted Averaging Filter
```
kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]])/16
weighted_average_filter = cv2.filter2D(image, -1, kernel)

plt.figure(figsize = (18, 6))
plt.subplot(121);plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122);plt.imshow(weighted_average_filter[:, :, ::-1]); plt.title('Output Image(weighted_average_filter)');plt.show()
```
#### Output
<img width="872" height="360" alt="image" src="https://github.com/user-attachments/assets/9dcff9b7-da2a-4813-bcd1-fb8c7ec41024" />


#### iii) Using Gaussian Filter
```
gaussian_filter = cv2.GaussianBlur(image, (29,29), 0, 0)

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(gaussian_filter[:, :, ::-1]); plt.title('Output Image ( Gaussian Filter)')
```
#### Output
<img width="874" height="359" alt="image" src="https://github.com/user-attachments/assets/896e474f-746a-4402-babf-5c28d321d927" />


#### iv)Using Median Filter
```
median_filter = cv2.medianBlur(image, 19)

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(median_filter[:, :, ::-1]); plt.title('Output Image ( Median_filter)')
```
#### Output
<img width="874" height="357" alt="image" src="https://github.com/user-attachments/assets/3a3cd503-2795-4fda-a5af-a4873babf4f5" />


### 2. Sharpening Filters

#### i) Using Laplacian Linear Kernal
```
# i) Using Laplacian Kernel (Manual Kernel)
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
sharpened_laplacian_kernel = cv2.filter2D(image, -1, kernel = laplacian_kernel)

# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(sharpened_laplacian_kernel[:, :, ::-1]); plt.title('Output Image ( Laplacian_filter)')
```
#### Output
<img width="873" height="359" alt="image" src="https://github.com/user-attachments/assets/1ecd4b9c-a060-4ce8-accf-5cbf5c156257" />


#### ii) Using Laplacian Operator
```
# ii) Using Laplacian Operator (OpenCV built-in)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
laplacian_operator = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_operator = np.uint8(np.absolute(laplacian_operator))

# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(131); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(132); plt.imshow(gray_image, cmap='gray'); plt.title('Gray_image')
plt.subplot(133); plt.imshow(laplacian_operator,cmap='gray'); plt.title('Output Image ( Laplacian_filter)')
```
#### Output
<img width="876" height="286" alt="image" src="https://github.com/user-attachments/assets/121d5ae5-82a0-4171-a2ad-a65d382e624d" />


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
