import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import signal


plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (20, 10)

def create_chirp_image(size=512):
    """生成 Chirp 信号图像 (频率随空间距离增加的信号)"""
    x = np.linspace(0, 1, size)

    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    
    chirp_img = 0.5 * (1 + np.cos(2 * np.pi * 40 * R**2)) 
    return chirp_img

def create_checkerboard_image(size=512, squares=16):
    """生成棋盘格图像"""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
   
    checker = (np.sin(2 * np.pi * squares * X) > 0) ^ (np.sin(2 * np.pi * squares * Y) > 0)
    return checker.astype(float)

def downsample_simple(img, factor):
    """简单下采样 (直接抽取像素)"""
    return img[::factor, ::factor]

def downsample_with_filter(img, factor):
    """高斯滤波后下采样"""
   
    sigma = factor / 2.0
    blurred = gaussian_filter(img, sigma=sigma)
    return blurred[::factor, ::factor], blurred

def plot_fft(img, title):
    """计算并绘制 FFT 频谱 (中心化)"""
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log(np.abs(f_shift) + 1e-6) # 取对数以便观察
    return magnitude


IMAGE_SIZE = 512
DOWNSAMPLE_FACTOR = 4
TARGET_SIZE = IMAGE_SIZE // DOWNSAMPLE_FACTOR


img_chirp = create_chirp_image(IMAGE_SIZE)
img_checker = create_checkerboard_image(IMAGE_SIZE, squares=16)



ds_chirp_direct = downsample_simple(img_chirp, DOWNSAMPLE_FACTOR)
ds_checker_direct = downsample_simple(img_checker, DOWNSAMPLE_FACTOR)


ds_chirp_filtered, blur_chirp = downsample_with_filter(img_chirp, DOWNSAMPLE_FACTOR)
ds_checker_filtered, blur_checker = downsample_with_filter(img_checker, DOWNSAMPLE_FACTOR)


fft_original = plot_fft(img_chirp, "Original")
fft_direct = plot_fft(ds_chirp_direct, "Direct Downsample")
fft_filtered = plot_fft(ds_chirp_filtered, "Filtered Downsample")


fig, axes = plt.subplots(3, 3, figsize=(18, 18))


axes[0, 0].imshow(img_chirp, cmap='gray')
axes[0, 0].set_title("1. 原始 Chirp 图像 (高分辨率)")
axes[0, 0].axis('off')

axes[0, 1].imshow(ds_chirp_direct, cmap='gray')
axes[0, 1].set_title(f"2. 直接下采样 (x{DOWNSAMPLE_FACTOR})\n注意：严重的莫尔条纹(混叠)")
axes[0, 1].axis('off')

axes[0, 2].imshow(ds_chirp_filtered, cmap='gray')
axes[0, 2].set_title(f"3. 高斯滤波后下采样 (x{DOWNSAMPLE_FACTOR})\n注意：边缘模糊但无混叠")
axes[0, 2].axis('off')


axes[1, 0].imshow(img_checker, cmap='gray')
axes[1, 0].set_title("4. 原始棋盘格图像")
axes[1, 0].axis('off')

axes[1, 1].imshow(ds_checker_direct, cmap='gray')
axes[1, 1].set_title("5. 直接下采样\n注意：边缘出现锯齿和波纹")
axes[1, 1].axis('off')

axes[1, 2].imshow(ds_checker_filtered, cmap='gray')
axes[1, 2].set_title("6. 高斯滤波后下采样\n注意：边缘平滑")
axes[1, 2].axis('off')


im0 = axes[2, 0].imshow(fft_original, cmap='viridis')
axes[2, 0].set_title("7. 原始图像频谱\n(高频信息丰富)")
axes[2, 0].axis('off')
plt.colorbar(im0, ax=axes[2, 0], fraction=0.046, pad=0.04)

im1 = axes[2, 1].imshow(fft_direct, cmap='viridis')
axes[2, 1].set_title("8. 直接下采样频谱\n(高频混叠：中心出现虚假频率分量)")
axes[2, 1].axis('off')
plt.colorbar(im1, ax=axes[2, 1], fraction=0.046, pad=0.04)

im2 = axes[2, 2].imshow(fft_filtered, cmap='viridis')
axes[2, 2].set_title("9. 滤波后下采样频谱\n(高频被截断：中心干净，无混叠)")
axes[2, 2].axis('off')
plt.colorbar(im2, ax=axes[2, 2], fraction=0.046, pad=0.04)

plt.suptitle("下采样混叠与抗混叠滤波对比实验", fontsize=16)
plt.tight_layout()
plt.show()