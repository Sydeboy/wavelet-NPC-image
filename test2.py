import pywt
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Decompose image using DWT
coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

# Show decomposition results
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(cA, cmap=plt.cm.gray)
ax[0].set_title('Approximation Coefficients')

ax[1].imshow(cH, cmap=plt.cm.gray)
ax[1].set_title('Horizontal Detail Coefficients')

ax[2].imshow(cV, cmap=plt.cm.gray)
ax[2].set_title('Vertical Detail Coefficients')

ax[3].imshow(cD, cmap=plt.cm.gray)
ax[3].set_title('Diagonal Detail Coefficients')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()
