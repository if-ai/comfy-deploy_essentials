# comfy-deploy_essentials

A collection of essential nodes for ComfyUI to enhance your workflow, focusing on stable image processing and utility functions.

## Features

### Image Processing
- **Pixelize Image**: Transform images into pixel art styles using the pixeloe library with adjustable parameters for downscaling mode, target size, patch size, thickness, and color matching.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/if-ai/comfy-deploy_Essentials.git
   ```

2. Install the required dependencies:
   ```bash
   pip install pixeloe
   ```

3. Restart ComfyUI

## Dependencies
- pixeloe: Required for the PixelOE node

## Usage

After installation, the nodes will appear in the ComfyUI menu under the "🔗ComfyDeploy" category.

## License

MIT
