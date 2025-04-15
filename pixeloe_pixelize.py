#pixeloe_pixelize.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
from nodes import MAX_RESOLUTION
import comfy.utils

class ComfyDeployPixelOE:
    """
    Applies pixelization effect using the pixeloe library, mimicking the
    functionality of the Essentials PixelOEPixelize node, including downscaling.
    Designed for the ComfyDeploy custom node set.
    Uses internal import for pixeloe library.
    """
    @classmethod
    def INPUT_TYPES(s):
        # Define standard downscale modes supported by pixeloe
        # We define these even if the import fails later, so the node *appears*
        # correctly initially, but will error during execution if pixeloe is missing.
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

        # --- Import pixeloe HERE, inside the execution method ---
        try:
            from pixeloe.pixelize import pixelize
        except ImportError:
            # This error will now happen at runtime if the library is missing
            raise ImportError("pixeloe library is required for the ComfyDeployPixelOE node, but it could not be imported. Please ensure it's installed in the ComfyDeploy environment.")
        # --- End of internal import ---

        batch_size = image.shape[0]
        output_images = []

        pbar = comfy.utils.ProgressBar(batch_size)
        for i in range(batch_size):
            img_tensor_single = image[i:i+1]
            img_np_uint8 = img_tensor_single.clone().mul(255).clamp(0, 255).byte().cpu().numpy()[0]

            try:
                pixelized_np = pixelize(
                    img_np_uint8,
                    mode=downscale_mode,
                    target_size=target_size,
                    patch_size=patch_size,
                    thickness=thickness,
                    contrast=1.0,
                    saturation=1.0,
                    color_matching=color_matching,
                    no_upscale=not output_original_size
                )
            except Exception as e:
                print(f"Error during pixeloe.pixelize execution: {e}")
                raise e # Re-raise the error after printing

            processed_img_tensor = T.ToTensor()(pixelized_np)
            processed_img_tensor = processed_img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
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