import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel


def create_complex_scene(size=512):
   
    img = np.zeros((size, size))
    
    for i in range(0, size, 32):
        for j in range(0, size//2, 32):
             img[i:i+16, j:j+16] = 1.0
             img[i+16:i+32, j+16:j+32] = 1.0
    
    x = np.linspace(0, 1, size//2)
    img[:, size//2:] = np.tile(x, (size, 1))
    return img

original = create_complex_scene()
M = 4


grad_x = sobel(original, axis=1)
grad_y = sobel(original, axis=0)

gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)


sigma_min, sigma_max = 0.5, 4.0

grad_norm = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)

sigma_map = sigma_min + grad_norm * (sigma_max - sigma_min)




img_uniform = gaussian_filter(original, sigma=1.8)

down_uniform = img_uniform[::M, ::M]


plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Scene\n(High Freq Left, Low Freq Right)")
plt.imshow(original, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gradient Magnitude\n(Estimated Local M)")
plt.imshow(gradient_magnitude, cmap='magma')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"Uniform Downsample (Sigma=1.8)\nResult Size: {down_uniform.shape}")
plt.imshow(down_uniform, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


print("分析：")
print("1. 左图是原始场景，左边是高频棋盘，右边是低频斜坡。")
print("2. 中图是梯度分析。可以看到左边的棋盘区域非常亮（值大），右边很暗（值小）。")
print("3. 根据图片要求，如果做自适应滤波：")
print("   - 左边（亮区）应该用大 sigma (接近 4.0) 强力模糊，防止棋盘混叠。")
print("   - 右边（暗区）应该用小 sigma (接近 0.5) 保持锐利。")
print("4. 右图是统一 sigma=1.8 的结果。")
print("   - 缺点：右边的斜坡可能变得过模糊了（损失了清晰度），而左边的棋盘可能还不够模糊。")