#pixeloe_pixelize.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
from nodes import MAX_RESOLUTION
import comfy.utils

# Attempt to import the core pixelize function from the pixeloe library
try:
    from pixeloe.pixelize import pixelize
    pixeloe_available = True
except ImportError:
    print("Warning: pixeloe library not found. ComfyDeployPixelOE node will not work.")
    pixeloe_available = False

class ComfyDeployPixelOE:
    """
    Applies pixelization effect using the pixeloe library, mimicking the
    functionality of the Essentials PixelOEPixelize node, including downscaling.
    Designed for the ComfyDeploy custom node set.
    """
    @classmethod
    def INPUT_TYPES(s):
        # Define standard downscale modes supported by pixeloe
        downscale_modes = ["contrast", "bicubic", "nearest", "center", "k-centroid"]
        
        # Check if pixeloe is available before defining inputs
        if not pixeloe_available:
             return {"required": {"error": ("STRING", {"default": "pixeloe library not found. Please install it.", "multiline": True})}}

        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_mode": (downscale_modes, {"default": "contrast"}),
                "target_size": ("INT", {
                    "default": 64,
                    "min": 4,
                    "max": 2048, # Adjusted max based on practical pixel art sizes
                    "step": 8,
                    "tooltip": "Target size (width/height) for the pixelated output BEFORE potential upscaling."
                }),
                "patch_size": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64, # Adjusted max
                    "step": 2,
                    "tooltip": "Size of the patches used for pixelization effect."
                }),
                "thickness": ("INT", {
                    "default": 2,
                    "min": 0, # Thickness can be 0
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
    CATEGORY = "🔗ComfyDeploy/Image Processing" # Added subcategory

    def pixelize_image(self, image, downscale_mode, target_size, patch_size, thickness, color_matching, output_original_size):
        
        if not pixeloe_available:
            raise ImportError("pixeloe library is required for the ComfyDeployPixelOE node.")

        # Convert ComfyUI IMAGE tensor (B, H, W, C, float 0-1) to NumPy array (H, W, C, uint8 0-255) for pixeloe
        # We process images in the batch individually
        
        batch_size = image.shape[0]
        output_images = []

        pbar = comfy.utils.ProgressBar(batch_size)
        for i in range(batch_size):
            img_tensor_single = image[i:i+1] # Keep batch dim for consistency, process one image
            
            # Convert single image tensor to NumPy array (H, W, C, uint8)
            img_np_uint8 = img_tensor_single.clone().mul(255).clamp(0, 255).byte().cpu().numpy()[0] # Remove batch dim for pixeloe

            # Call the pixeloe pixelize function
            try:
                pixelized_np = pixelize(
                    img_np_uint8,
                    mode=downscale_mode,
                    target_size=target_size,
                    patch_size=patch_size,
                    thickness=thickness,
                    contrast=1.0,  # Hardcoded 
                    saturation=1.0, # Hardcoded 
                    color_matching=color_matching,
                    no_upscale=not output_original_size # Inverse logic: if we want original size, no_upscale=False
                )
            except Exception as e:
                print(f"Error during pixeloe.pixelize: {e}")
                # Return the original image slice on error? Or raise? Let's raise for now.
                raise e

            # Convert the result back to ComfyUI IMAGE tensor format (B, H, W, C, float 0-1)
            # T.ToTensor converts (H, W, C) uint8 [0-255] -> (C, H, W) float [0-1]
            processed_img_tensor = T.ToTensor()(pixelized_np)
            
            # Convert (C, H, W) float [0-1] -> (B, H, W, C) float [0-1]
            processed_img_tensor = processed_img_tensor.unsqueeze(0).permute(0, 2, 3, 1)

            output_images.append(processed_img_tensor)
            pbar.update(1)

        # Combine processed images back into a single batch tensor
        if not output_images:
             # Should not happen if input batch > 0, but handle defensively
             return (torch.empty_like(image[0:0]),) # Return empty tensor with correct dtype/device
             
        final_output_tensor = torch.cat(output_images, dim=0)

        return (final_output_tensor,)

# Add mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ComfyDeployPixelOE": ComfyDeployPixelOE
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyDeployPixelOE": "Pixelize Image (ComfyDeploy)"
}