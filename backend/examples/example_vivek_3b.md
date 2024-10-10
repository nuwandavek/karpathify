## Lesson 1: Introduction to Monocular Depth Estimation and Vision Transformers

Understand the basics of monocular depth estimation and how Vision Transformers (ViT) can be applied to this problem.

**Note:** We begin by importing necessary modules. The `DepthPro` class inherits from `nn.Module`, the base class for all neural network modules in PyTorch.

```python
# depth_pro.py
import torch
from torch import nn

class DepthPro(nn.Module):
    def __init__(self, encoder, decoder, last_dims, use_fov_head=True, fov_encoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # ...
```

**Note:** Monocular depth estimation aims to predict depth maps using only one image, without stereo vision cues. Machine learning models learn to infer depth from visual cues in single images.

```python
# Understanding Monocular Depth Estimation
# Depth estimation from a single image involves predicting depth for each pixel.
# This is challenging due to the lack of stereo information.
```

**Note:** Vision Transformers adapt the transformer architecture, originally designed for natural language processing, to image data. They split images into patches and process them similarly to words in a sentence.

```python
# Introducing Vision Transformers (ViT)
from timm import create_model

vit_model = create_model('vit_base_patch16_224', pretrained=True)
```

**Note:** The `create_vit` function builds a ViT model based on a preset configuration. The `timm` library provides pre-trained models and utilities for building vision models.

```python
# vit_factory.py
def create_vit(preset, use_pretrained=False, checkpoint_uri=None, use_grad_checkpointing=False):
    config = VIT_CONFIG_DICT[preset]
    model = timm.create_model(
        config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
    )
    return model.model
```

**Note:** ViT processes image patches as a sequence, similar to tokens in NLP models. This allows the model to capture global context across the entire image.

```python
# Exploring the ViT Architecture
# ViT models split the image into patches and linearly embed each patch.
# Positional embeddings are added, and the sequence is fed into transformer layers.
```

## Lesson 2: Exploring the Multi-Scale ViT Architecture in Depth Pro

Learn how Depth Pro applies ViT to multi-scale patches and how the encoder processes inputs.

**Note:** The `DepthProEncoder` class handles the encoding of input images at multiple scales. It uses both a patch encoder and an image encoder to capture local and global features.

```python
# encoder.py
class DepthProEncoder(nn.Module):
    def __init__(self, dims_encoder, patch_encoder, image_encoder, hook_block_ids, decoder_features):
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        # ...
```

**Note:** By processing the image at different scales, the model can focus on fine-grained details at higher resolutions and broader structures at lower resolutions.

```python
# Creating a Multi-Scale Image Pyramid
# The encoder creates an image pyramid to process the image at multiple resolutions.
# This helps in capturing both fine details and global context.
```

**Note:** The `split` function divides the image into overlapping patches. Overlapping helps in reducing edge artifacts and ensuring continuity between patches.

```python
# Splitting the Image into Overlapping Patches
def split(self, x, overlap_ratio=0.25):
    # Split input into patches with a sliding window mechanism
    # ...
```

**Note:** Each patch is encoded using the `patch_encoder`, which is a ViT model. The features from all patches are then merged to form a coherent representation of the image.

```python
# Processing Patches with the ViT Encoder
patch_features = self.patch_encoder(patch_images)
# Reshape and merge features from patches
```

**Note:** The encoder outputs features at multiple resolutions. These features are essential for the decoder to produce high-resolution depth maps with fine details.

```python
# Combining Features from Multiple Scales
# After encoding at different scales, features are combined for the decoder.
# This fusion of multi-scale features is key to producing detailed depth maps.
```

## Lesson 3: Decoding Multi-Scale Features into High-Resolution Depth Maps

Understand how the decoder assembles encoded features to produce the final depth map.

**Note:** The `MultiresConvDecoder` class is responsible for decoding the multi-scale features into a high-resolution depth map. It uses convolutional layers to process and upsample features.

```python
# decoder.py
class MultiresConvDecoder(nn.Module):
    def __init__(self, dims_encoder, dim_decoder):
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.dim_decoder = dim_decoder
        # ...
```

**Note:** A 1x1 convolution is used to project features from the encoder to the decoder's expected dimensionality. This ensures compatibility between encoder and decoder features.

```python
# Projecting Encoder Features
conv0 = nn.Conv2d(self.dims_encoder[0], dim_decoder, kernel_size=1, bias=False)
# Apply convolution to match dimensions
```

**Note:** Feature fusion blocks combine features from different scales and stages of the encoder. They help in integrating information across scales.

```python
# Feature Fusion Blocks
self.fusions = nn.ModuleList([
    FeatureFusionBlock2d(dim_decoder) for _ in range(num_encoders)
])
```

**Note:** The decoder processes features in a bottom-up manner, starting from the lowest resolution and progressively integrating higher-resolution features.

```python
# decoder.py
def forward(self, encodings):
    # Decode features starting from the lowest resolution
    features = self.convs[-1](encodings[-1])
    features = self.fusions[-1](features)
    # ...
```

**Note:** After decoding and upsampling, the model produces the canonical inverse depth map. Inverse depth emphasizes closer objects, which are often more important in applications.

```python
# Generating the Depth Map
canonical_inverse_depth = self.head(features)
# Apply activation to obtain the final depth predictions
```

## Lesson 4: Focal Length Estimation without Camera Intrinsics

Learn how Depth Pro estimates the focal length directly from the image to produce metric depth maps.

**Note:** The `FOVNetwork` class estimates the field of view (FoV) of the input image, which is used to calculate the focal length.

```python
# fov.py
class FOVNetwork(nn.Module):
    def __init__(self, num_features, fov_encoder=None):
        super().__init__()
        # Initialize layers for estimating field of view
        # ...
```

**Note:** By estimating the FoV, the model calculates the focal length in pixels. This allows it to produce metric depth estimates without provided camera intrinsics.

```python
# Estimating Focal Length
fov_deg = self.fov.forward(x, features_0.detach())
# Convert FoV to focal length in pixels
f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
```

**Note:** During inference, if the focal length is not provided, the model uses the estimated FoV to compute it. This ensures that depth predictions have the correct scale.

```python
# depth_pro.py
@torch.no_grad()
def infer(self, x, f_px=None, interpolation_mode='bilinear'):
    # ...
    if f_px is None:
        f_px = computed from estimated FoV
    inverse_depth = canonical_inverse_depth * (W / f_px)
    # ...
```

**Note:** The model adjusts the canonical inverse depth using the focal length to obtain the final metric depth map. Clamping ensures numerical stability.

```python
# Combining Depth and Focal Length
inverse_depth = canonical_inverse_depth * (W / f_px)

depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
```

**Note:** Estimating focal length directly from images allows Depth Pro to work with any image, regardless of whether camera metadata is available.

```python
# Handling Images without Metadata
# The ability to estimate focal length makes the model robust to images lacking EXIF data.
# This is crucial for processing in-the-wild images.
```

## Lesson 5: Integrating Components and Understanding the Full Depth Pro Model

Review the complete Depth Pro model, understand the training protocol, and explore the evaluation metrics.

**Note:** The `DepthPro` class integrates the encoder, decoder, and focal length estimation components. The `head` processes decoder outputs to produce the final depth map.

```python
# depth_pro.py
class DepthPro(nn.Module):
    def __init__(self, encoder, decoder, last_dims, use_fov_head=True, fov_encoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Initialize head for depth prediction
        self.head = nn.Sequential(
            nn.Conv2d(...),
            # ...
        )
```

**Note:** Training begins with mixed datasets to learn robust features, followed by fine-tuning on high-quality synthetic data to improve boundary sharpness.

```python
# Training Protocol
# The model is trained using a curriculum that starts with real-world datasets
# and fine-tunes on synthetic datasets with accurate ground truth.
```

**Note:** The loss functions include mean absolute error (MAE) and terms that encourage sharp gradients and fine details in the depth map.

```python
# Loss Functions
# The training uses losses on the depth values and their derivatives:
loss = L_MAE + L_MAGE + L_MALE + L_MSGE
```

**Note:** These metrics focus on the accuracy of depth boundaries, particularly around thin structures and object edges, which are critical for applications like view synthesis.

```python
# Evaluation Metrics
# Custom metrics are introduced to assess boundary accuracy:
# - Edge recall
# - Boundary F1 score
```

**Note:** An example of running inference with Depth Pro. The `infer` method handles preprocessing, prediction, and post-processing to obtain the depth map.

```python
# Inference Example
# cli/run.py
image, _, _ = load_rgb(image_path)
prediction = model.infer(transform(image))
depth = prediction['depth'].detach().cpu().numpy()
```

