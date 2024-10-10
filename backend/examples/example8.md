## Lesson 1: Introduction to DepthPro and Monocular Depth Estimation

In this lesson, we will explore the main `DepthPro` class and understand how it performs monocular depth estimation.

**Note:** We start by importing the necessary PyTorch modules. The `torch.nn` module provides neural network components.

```python
import torch
import torch.nn as nn
```

**Note:** The `DepthPro` class is the main model for depth estimation. It inherits from `nn.Module`, which is the base class for all neural network modules in PyTorch.

```python
class DepthPro(nn.Module):
    def __init__(self, encoder, decoder, last_dims, use_fov_head=True, fov_encoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
```

**Note:** The `forward` method defines how the input `x` is transformed through the encoder and decoder to produce the depth map.

```python
def forward(self, x):
    _, _, H, W = x.shape
    assert H == self.img_size and W == self.img_size
    
    encodings = self.encoder(x)
    features, features_0 = self.decoder(encodings)
    canonical_inverse_depth = self.head(features)
```

**Note:** The `infer` method performs inference on an input image `x`. It handles resizing and computes the metric depth using the estimated focal length if `f_px` is not provided.

```python
def infer(self, x, f_px=None, interpolation_mode="bilinear"):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    _, _, H, W = x.shape
    resize = H != self.img_size or W != self.img_size
    
    if resize:
        x = nn.functional.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode=interpolation_mode,
            align_corners=False
        )
    
    canonical_inverse_depth, fov_deg = self.forward(x)
    if f_px is None:
        f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
    
    inverse_depth = canonical_inverse_depth * (W / f_px)
    depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
```

**Note:** In the paper (Section 3.1), the authors describe how DepthPro uses Vision Transformers (ViT) at multiple scales to capture both global context and fine details.

```python
# From the paper, Section 3.1 Network
# The DepthPro architecture applies a plain ViT encoder at multiple scales and fuses the patch predictions into a single high-resolution output.
```

**Note:** This formula, mentioned in the paper, shows how DepthPro computes metric depth from the canonical inverse depth and the estimated focal length.

```python
# Equation from the paper:
# D_m = f_px / (w * C)
# where D_m is the metric depth, f_px is the focal length in pixels, w is the image width, and C is the canonical inverse depth.
```

## Lesson 2: Understanding the DepthPro Encoder

In this lesson, we delve into the encoder architecture of DepthPro, focusing on how it extracts multi-resolution features using Vision Transformers.

**Note:** The `DepthProEncoder` class combines a patch encoder and an image encoder to create multi-resolution encodings.

```python
# encoder.py
class DepthProEncoder(nn.Module):
    def __init__(self, dims_encoder, patch_encoder, image_encoder, hook_block_ids, decoder_features):
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
```

**Note:** In DepthPro, the patch encoder captures local details by processing patches, while the image encoder captures the overall scene.

```python
# The patch encoder processes overlapping patches at multiple scales.
# The image encoder provides global context.
```

**Note:** The `_create_pyramid` method creates a 3-level image pyramid to process the image at different scales.

```python
def _create_pyramid(self, x):
    x0 = x
    x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
    return x0, x1, x2
```

**Note:** In the `forward` method, the encoder splits the image into patches at different scales to capture multi-scale features.

```python
def forward(self, x):
    x0, x1, x2 = self._create_pyramid(x)
    # Process patches at multiple scales
    x0_patches = self.split(x0, overlap_ratio=0.25)
    x1_patches = self.split(x1, overlap_ratio=0.5)
    x2_patches = x2
```

**Note:** The paper explains that after processing patches with the ViT encoders, features are merged and upsampled to produce high-resolution feature maps.

```python
# From the paper, Section 3.1 Network
# "Patches are merged into feature maps, upsampled, and fused via a DPT decoder."

```

**Note:** The `create_vit` function builds a Vision Transformer model that serves as the backbone for the encoder.

```python
# vit_factory.py
from .vit import make_vit_b16_backbone

def create_vit(preset, use_pretrained=False, ...):
    # Creates and loads a ViT backbone module
    model = timm.create_model(
        config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
    )
    model = make_vit_b16_backbone(model, ...)
```

**Note:** The authors highlight the advantage of using standard ViT models, which allows leveraging pretrained models for better performance.

```python
# From the paper, Section 3.1 Network
# "A key benefit of assembling our architecture from plain ViT encoders over custom encoders is the abundance of pretrained ViT-based backbones that can be harnessed."
```

## Lesson 3: Exploring the Decoder and Multiresolution Fusion

In this lesson, we study the decoder module, and how it fuses features to produce high-resolution depth maps.

**Note:** The `MultiresConvDecoder` class takes encoder features from multiple resolutions and decodes them into a depth map.

```python
# decoder.py
class MultiresConvDecoder(nn.Module):
    def __init__(self, dims_encoder, dim_decoder):
        super().__init__()
        self.convs = nn.ModuleList([...])
        self.fusions = nn.ModuleList([...])
```

**Note:** In the `forward` method, features from different encoder levels are fused together using `FeatureFusionBlock2d`.

```python
def forward(self, encodings):
    features = self.convs[-1](encodings[-1])
    features = self.fusions[-1](features)
    for i in range(len(encodings) - 2, -1, -1):
        features_i = self.convs[i](encodings[i])
        features = self.fusions[i](features, features_i)
    return features
```

**Note:** The `FeatureFusionBlock2d` class fuses features from different resolutions and includes residual blocks to learn refinements.

```python
class FeatureFusionBlock2d(nn.Module):
    def __init__(self, num_features, deconv=False, batch_norm=False):
        super().__init__()
        self.resnet1 = self._residual_block(num_features, batch_norm)
        self.resnet2 = self._residual_block(num_features, batch_norm)
        # ...
```

**Note:** Residual blocks help in training deep networks by allowing gradients to flow through skip connections.

```python
def _residual_block(num_features, batch_norm):
    # Creates a residual block with optional batch normalization
    layers = [...]
```

**Note:** The decoder combines the overlapping patches processed by the encoder to produce seamless high-resolution outputs.

```python
# From the paper, Section 3.1 Network
# "After downsampling to 1536×1536, the input image is split into patches of 384×384. For the two finest scales, we let patches overlap to avoid seams."

```

## Lesson 4: Field of View Estimation in DepthPro

In this lesson, we explore how DepthPro estimates the focal length from a single image using a Field of View (FoV) network.

**Note:** The `FOVNetwork` estimates the field of view (FoV) of the camera, which is necessary for computing metric depth without camera intrinsics.

```python
# fov.py
class FOVNetwork(nn.Module):
    def __init__(self, num_features, fov_encoder=None):
        super().__init__()
        # Define the FOV head layers
        self.head = nn.Sequential(...)
```

**Note:** The `forward` method combines features from the input image and the low-resolution feature map to estimate the FoV.

```python
def forward(self, x, lowres_feature):
    if hasattr(self, 'encoder'):
        x = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x = self.encoder(x)
        lowres_feature = self.downsample(lowres_feature)
        x = x.reshape_as(lowres_feature) + lowres_feature
    else:
        x = lowres_feature
    return self.head(x)
```

**Note:** Estimating the focal length allows DepthPro to compute metric depth even when camera intrinsics are unavailable.

```python
# From the paper, Section 3.3 Focal Length Estimation
# "To handle images that may have inaccurate or missing EXIF metadata, we supplement our network with a focal length estimation head."
```

**Note:** This formula shows how the estimated FoV in degrees is converted to focal length in pixels.

```python
# Equation from the paper:
# f_px = 0.5 * w / tan(0.5 * FoV_deg * π / 180)
# where f_px is the focal length in pixels, w is the image width, and FoV_deg is the estimated field of view in degrees.
```

## Lesson 5: Running DepthPro for Inference and Applications

In this final lesson, we will learn how to use the DepthPro model for inference on images and understand how to integrate it into applications like novel view synthesis.

**Note:** The `run.py` script provides an example of how to run DepthPro on a sample image using command-line arguments.

```python
# run.py
import argparse
from depth_pro import create_model_and_transforms, load_rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ...
    args = parser.parse_args()
```

**Note:** We create the DepthPro model and set it to evaluation mode. The `create_model_and_transforms` function sets up the model and necessary image transformations.

```python
model, transform = create_model_and_transforms(device=get_torch_device(), precision=torch.half)
model.eval()
```

**Note:** We load an image and its focal length (if available), then perform inference to obtain the depth map.

```python
image, _, f_px = load_rgb(image_path)
prediction = model.infer(transform(image), f_px=f_px)
```

**Note:** DepthPro can estimate depth maps with absolute scale even when camera intrinsics are missing, enabling applications like novel view synthesis.

```python
# From the paper, Section 1 Introduction
# "Our model, DepthPro, produces metric depth maps with absolute scale on arbitrary images ‘in the wild’ without requiring metadata such as camera intrinsics."
```

**Note:** We can visualize the predicted depth map using a colormap for better interpretation.

```python
# Example of visualizing the depth map
import matplotlib.pyplot as plt
plt.imshow(prediction['depth'], cmap='turbo')
plt.show()
```

**Note:** Depth maps estimated by DepthPro can be used in applications such as synthesizing new viewpoints from a single image.

```python
# Application: Novel View Synthesis
# Using the estimated depth map, we can generate new views of the scene by warping the image.
```


