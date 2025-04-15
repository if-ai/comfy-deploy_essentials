#pixeloe_pixelize.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
from nodes import MAX_RESOLUTION
import comfy.utils

class ComfyDeployPixelOE:
    """
    Applies pixelization effect using the pixeloe library
    """
    @classmethod
    def INPUT_TYPES(s):
        # Define standard downscale modes supported by pixeloe
        downscale_modes = ["contrast", "bicubic", "nearest", "center", "k-centroid"]

        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_mode": (downscale_modes, {"default": "contrast"}),
                "target_size": ("INT", {
                    "default": 64,
                    "min": 4,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Target size (width/height) for the pixelated output BEFORE potential upscaling."
                }),
                "patch_size": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 2,
                    "tooltip": "Size of the patches used for pixelization effect."
                }),
                "thickness": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Thickness of added outlines/edges."
                }),
                "color_matching": ("BOOLEAN", {"default": True}),
                "do_quant": ("BOOLEAN", {"default": True}),
                "output_original_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, upscale the pixelized image back to the original input size. If False, output at target_size."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pixelize_image"
    CATEGORY = "🔗ComfyDeploy/Image Processing"

    def pixelize_image(self, image, downscale_mode, target_size, patch_size, thickness, color_matching, do_quant, output_original_size):
        
        try:
            # Import the PyTorch-specific pixelize function
            from pixeloe.torch.pixelize import pixelize_pytorch
        except ImportError as e:
            # Custom implementation of pixelize_pytorch if import fails
            # This is a simplified version of the original function for compatibility
            def pixelize_pytorch(img_t, target_size=256, patch_size=6, thickness=3, mode="contrast", 
                                do_color_match=True, do_quant=False, K=32):
                """
                Fallback implementation of pixelize_pytorch for compatibility
                """
                import torch.nn.functional as F
                
                # Simple nearest-neighbor downscaling followed by upscaling to create pixelization effect
                C, H, W = img_t.shape
                ratio = W / H
                out_h = int((target_size**2 / ratio) ** 0.5)
                out_w = int(out_h * ratio)
                
                # Downscale
                down = F.interpolate(
                    img_t.unsqueeze(0), size=(out_h, out_w), mode="nearest"
                )[0]
                
                # Upscale with nearest neighbor to get pixelation effect
                out_pixel = F.interpolate(
                    down.unsqueeze(0), scale_factor=patch_size, mode="nearest"
                )[0]
                
                return out_pixel
            
            print(f"Using fallback implementation for pixelize_pytorch due to: {e}")

        batch_size = image.shape[0]
        output_images = []

        pbar = comfy.utils.ProgressBar(batch_size)
        for i in range(batch_size):
            # Get single image tensor from batch (keep in tensor format for PyTorch version)
            img_tensor_single = image[i:i+1][0]  # shape: [H, W, C]
            
            # The PyTorch version expects [C, H, W] format
            img_tensor_chw = img_tensor_single.permute(2, 0, 1)
            
            try:
                # Use the EXACT parameter names from the function definition you shared
                pixelized_tensor = pixelize_pytorch(
                    img_t=img_tensor_chw,
                    target_size=target_size,
                    patch_size=patch_size,
                    thickness=thickness,
                    mode=downscale_mode,
                    do_color_match=color_matching,
                    do_quant=do_quant,
                    K=32
                    )
                
                # Handle output size if needed
                if not output_original_size:
                    # Check current size vs target_size
                    current_h, current_w = pixelized_tensor.shape[1:3]
                    if current_h != target_size or current_w != target_size:
                        # Resize to target size using the same interpolation as would be used for downscaling
                        mode = 'bilinear'  # Default mode
                        if downscale_mode == 'nearest':
                            mode = 'nearest'
                        elif downscale_mode == 'bicubic':
                            mode = 'bicubic'
                            
                        pixelized_tensor = torch.nn.functional.interpolate(
                            pixelized_tensor.unsqueeze(0),
                            size=(target_size, target_size),
                            mode=mode,
                            align_corners=False if mode != 'nearest' else None
                        ).squeeze(0)
                
                # Convert back to [H, W, C] format for ComfyUI
                processed_img_tensor = pixelized_tensor.permute(1, 2, 0)
                
                # Add batch dimension and clamp to valid range
                processed_img_tensor = processed_img_tensor.unsqueeze(0).clamp(0, 1)
                
            except Exception as e:
                print(f"Error during pixeloe.torch.pixelize execution: {e}")
                raise e

            output_images.append(processed_img_tensor)
            pbar.update(1)

        if not output_images:
             return (torch.empty_like(image[0:0]),)

        final_output_tensor = torch.cat(output_images, dim=0)
        return (final_output_tensor,)

# Add mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ComfyDeployPixelOE": ComfyDeployPixelOE
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyDeployPixelOE": "Pixelize Image (ComfyDeploy)"
}