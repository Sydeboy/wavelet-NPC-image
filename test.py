import pywt
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Decompose image using DWT
coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

# Show original image and decomposition results
fig, ax = plt.subplots(1, 5, figsize=(12, 6))

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original Image')

ax[1].imshow(cA, cmap=plt.cm.gray)
ax[1].set_title('Approximation Coefficients')

ax[2].imshow(cH, cmap=plt.cm.gray)
ax[2].set_title('Horizontal Detail Coefficients')

ax[3].imshow(cV, cmap=plt.cm.gray)
ax[3].set_title('Vertical Detail Coefficients')

ax[4].imshow(cD, cmap=plt.cm.gray)
ax[4].set_title('Diagonal Detail Coefficients')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()
