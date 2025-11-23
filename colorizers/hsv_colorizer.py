"""
HSV-based Colorization Module

This module implements colorization using HSV color space:
- Converts grayscale image to HSV
- Uses grayscale as V (value) channel
- Generates H (hue) and S (saturation) using rule-based patterns
- Converts back to RGB
"""

import numpy as np
from skimage import color


def colorize_hsv(img_rgb):
    """
    Colorize a grayscale or RGB image using HSV color space.
    
    Args:
        img_rgb: Input image as RGB numpy array (H, W, 3) with values in [0, 255]
                 or grayscale (H, W) with values in [0, 255]
    
    Returns:
        Colorized RGB image as numpy array (H, W, 3) with values in [0, 1]
    """
    # Normalize input to [0, 1] range if needed
    if img_rgb.max() > 1.0:
        img_rgb_normalized = img_rgb.astype(np.float32) / 255.0
    else:
        img_rgb_normalized = img_rgb.astype(np.float32)
    
    # Convert to grayscale if needed
    if img_rgb_normalized.ndim == 3:
        # Convert RGB to grayscale (rgb2gray expects [0, 1] range)
        img_gray = color.rgb2gray(img_rgb_normalized)
    else:
        img_gray = img_rgb_normalized.copy()
    
    H, W = img_gray.shape
    
    # Create HSV image
    # V (Value) channel: use grayscale image
    V = img_gray
    
    # H (Hue) channel: map intensity to hue using a gradient
    # Dark regions -> blue (0.6), bright regions -> yellow (0.15)
    # This creates a natural-looking color gradient
    H_channel = 0.6 - 0.45 * img_gray  # Blue to yellow gradient
    
    # S (Saturation) channel: higher saturation for mid-tones
    # Create a bell curve centered at 0.5 intensity
    S_channel = 4 * img_gray * (1 - img_gray)  # Parabolic curve, max at 0.5
    
    # Stack channels to create HSV image
    img_hsv = np.zeros((H, W, 3))
    img_hsv[:, :, 0] = H_channel  # Hue [0, 1]
    img_hsv[:, :, 1] = S_channel  # Saturation [0, 1]
    img_hsv[:, :, 2] = V  # Value [0, 1]
    
    # Convert HSV to RGB
    # skimage expects HSV in [0, 1] range, which we have
    img_rgb_colorized = color.hsv2rgb(img_hsv)
    
    # Ensure values are in [0, 1] range
    img_rgb_colorized = np.clip(img_rgb_colorized, 0, 1)
    
    return img_rgb_colorized


def colorize_hsv_advanced(img_rgb, hue_shift=0.0, saturation_boost=1.0):
    """
    Advanced HSV colorization with adjustable parameters.
    
    Args:
        img_rgb: Input image as RGB numpy array (H, W, 3) with values in [0, 255]
        hue_shift: Shift hue by this amount (0.0 to 1.0)
        saturation_boost: Multiply saturation by this factor
    
    Returns:
        Colorized RGB image as numpy array (H, W, 3) with values in [0, 1]
    """
    # Normalize input to [0, 1] range if needed
    if img_rgb.max() > 1.0:
        img_rgb_normalized = img_rgb.astype(np.float32) / 255.0
    else:
        img_rgb_normalized = img_rgb.astype(np.float32)
    
    # Convert to grayscale if needed
    if img_rgb_normalized.ndim == 3:
        img_gray = color.rgb2gray(img_rgb_normalized)
    else:
        img_gray = img_rgb_normalized.copy()
    
    H, W = img_gray.shape
    
    # V channel: grayscale
    V = img_gray
    
    # H channel: intensity-based hue mapping with shift
    H_channel = (0.6 - 0.45 * img_gray + hue_shift) % 1.0
    
    # S channel: enhanced saturation
    S_channel = np.clip(4 * img_gray * (1 - img_gray) * saturation_boost, 0, 1)
    
    # Create HSV image
    img_hsv = np.zeros((H, W, 3))
    img_hsv[:, :, 0] = H_channel
    img_hsv[:, :, 1] = S_channel
    img_hsv[:, :, 2] = V
    
    # Convert to RGB
    img_rgb_colorized = color.hsv2rgb(img_hsv)
    img_rgb_colorized = np.clip(img_rgb_colorized, 0, 1)
    
    return img_rgb_colorized

