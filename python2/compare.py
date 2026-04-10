import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def create_checkerboard(size=512, grid_size=16):
    img = np.zeros((size, size))
    square_size = size // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                img[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 1.0
    return img


original_img = create_checkerboard()
M = 4  


sigmas = [0.5, 1.0, 1.8, 2.0, 4.0]
titles = [f'Sigma = {s}' for s in sigmas]


processed_images = []

for sigma in sigmas:
   
    blurred = gaussian_filter(original_img, sigma=sigma)
    
    downsampled = blurred[::M, ::M]
    processed_images.append(downsampled)


plt.figure(figsize=(15, 8))


plt.subplot(2, 3, 1)
plt.imshow(original_img, cmap='gray', interpolation='nearest')
plt.title('Original High Res')
plt.axis('off')


for i, img in enumerate(processed_images):
    plt.subplot(2, 3, i + 2)
   
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title(f'Downsample M=4\nSigma = {sigmas[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()


print(f"理论最佳 Sigma (0.45 * M) = {0.45 * M}")
print("观察指南：")
print("- Sigma=0.5: 边缘仍有明显的锯齿和混叠（未充分滤波）。")
print("- Sigma=1.8: 边缘平滑，无混叠，且图像最清晰（理论最佳）。")
print("- Sigma=4.0: 图像过度模糊，细节丢失严重（过度滤波）。")