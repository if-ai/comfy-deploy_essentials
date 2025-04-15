from .comfy_nodes.pixeloe_pixelize import ComfyDeployPixelOE

NODE_CLASS_MAPPINGS = {
    "ComfyDeployPixelOE": ComfyDeployPixelOE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyDeployPixelOE": "Pixelize Image (ComfyDeploy)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("---------------------------------------------------")
print("- ComfyDeploy_Essentials: Loaded")
print("---------------------------------------------------")