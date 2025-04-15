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
                "output_original_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, upscale the pixelized image back to the original input size. If False, output at target_size."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pixelize_image"
    CATEGORY = "🔗ComfyDeploy/Image Processing"

    def pixelize_image(self, image, downscale_mode, target_size, patch_size, thickness, color_matching, output_original_size):
        
        try:
            # Import the PyTorch-specific pixelize function
            from pixeloe.torch.pixelize import pixelize as pixelize_fn
        except ImportError:
            raise ImportError("pixeloe library is required for the ComfyDeployPixelOE node, but it could not be imported. Please ensure it's installed in the ComfyDeploy environment.")

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
                pixelized_tensor = pixelize_fn(
                    img_t=img_tensor_chw,
                    target_size=target_size,
                    patch_size=patch_size,
                    thickness=thickness,
                    mode=downscale_mode,
                    do_color_match=color_matching
                    # Other params are optional and use defaults
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
