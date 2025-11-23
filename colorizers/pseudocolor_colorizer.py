"""
Pseudocolor Mapping Module

This module implements intensity-based colormap mapping for colorization.
Supports multiple colormap modes: Jet, Hot, Cool, Viridis, Parula, etc.
"""

import numpy as np
from matplotlib import cm
from skimage import color


# Available colormap modes
COLORMAP_MODES = {
    'jet': cm.jet,
    'hot': cm.hot,
    'cool': cm.cool,
    'viridis': cm.viridis,
    'parula': cm.viridis,  # Parula is similar to viridis, using viridis as substitute
    'plasma': cm.plasma,
    'inferno': cm.inferno,
    'magma': cm.magma,
    'spring': cm.spring,
    'summer': cm.summer,
    'autumn': cm.autumn,
    'winter': cm.winter,
    'rainbow': cm.rainbow,
    'turbo': cm.turbo,
    'hsv': cm.hsv,
    'seismic': cm.seismic,
    'terrain': cm.terrain,
}


def colorize_pseudocolor(img_rgb, colormap_mode='jet'):
    """
    Colorize a grayscale or RGB image using pseudocolor mapping.
    
    Args:
        img_rgb: Input image as RGB numpy array (H, W, 3) with values in [0, 255]
                 or grayscale (H, W) with values in [0, 255]
        colormap_mode: String specifying the colormap to use.
                      Options: 'jet', 'hot', 'cool', 'viridis', 'parula', 
                               'plasma', 'inferno', 'magma', 'spring', 'summer',
                               'autumn', 'winter', 'rainbow', 'turbo', 'hsv',
                               'seismic', 'terrain'
                      Default: 'jet'
    
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
    
    # Get colormap
    if colormap_mode.lower() not in COLORMAP_MODES:
        print(f"Warning: Unknown colormap '{colormap_mode}', using 'jet' instead.")
        colormap_mode = 'jet'
    
    colormap = COLORMAP_MODES[colormap_mode.lower()]
    
    # Apply colormap: map intensity [0, 1] to RGB colors
    # colormap returns RGBA, we only need RGB
    img_colored = colormap(img_gray)[:, :, :3]
    
    # Ensure values are in [0, 1] range
    img_colored = np.clip(img_colored, 0, 1)
    
    return img_colored


def colorize_pseudocolor_multiple(img_rgb, colormap_modes=None):
    """
    Apply multiple pseudocolor mappings to an image.
    
    Args:
        img_rgb: Input image as RGB numpy array (H, W, 3) with values in [0, 255]
        colormap_modes: List of colormap mode strings. If None, uses default set.
    
    Returns:
        Dictionary mapping colormap names to colorized images (all in [0, 1] range)
    """
    if colormap_modes is None:
        colormap_modes = ['jet', 'hot', 'viridis', 'plasma']
    
    results = {}
    for mode in colormap_modes:
        results[mode] = colorize_pseudocolor(img_rgb, mode)
    
    return results


def get_available_colormaps():
    """
    Get list of available colormap modes.
    
    Returns:
        List of available colormap mode strings
    """
    return list(COLORMAP_MODES.keys())

