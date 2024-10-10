# Lesson 1: Introduction to Depth Estimation and Vision Transformers

**Concepts Covered:**
- Monocular depth estimation
- Vision Transformers (ViT)
  
**Overview:**

In this lesson, we'll introduce the basics of monocular depth estimation and how Vision Transformers (ViT) are applied in this context. We'll explore how ViTs can be used for dense prediction tasks like depth estimation, setting the stage for understanding the DepthPro model.

**Further Reading:**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)

**Code Explanation:**

We'll start by loading a pre-trained ViT model using PyTorch and process an input image to extract features. This will help you understand how ViT can serve as an encoder in depth estimation models.

**Code:**

```python
import torch
import torchvision.transforms as T
from PIL import Image
from timm import create_model

# Load a pre-trained ViT model
model_name = 'vit_base_patch16_224'
model = create_model(model_name, pretrained=True)
model.eval()

# Image preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=0.5, std=0.5),
])

# Load and preprocess the image
image = Image.open('sample_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Pass the image through the ViT model
with torch.no_grad():
    features = model.forward_features(input_tensor)

print("Extracted features shape:", features.shape)
```

**Notes:**

- We use the `timm` library to load a pre-trained ViT model.
- The `forward_features` method extracts features from the input image.
- These features can be used in downstream tasks like depth estimation.

**What to Read Next:**

- Dive deeper into how transformers work in computer vision: [A Survey on Visual Transformer](https://arxiv.org/abs/2012.12556)
- Understand the role of feature extraction in depth estimation.

# Lesson 2: Understanding the DepthPro Encoder

**Concepts Covered:**
- Multi-scale image processing
- Overlapping patches and ViT encoders
- DepthProEncoder architecture

**Overview:**

In this lesson, we'll explore how the DepthPro model processes images at multiple scales using the `DepthProEncoder`. We'll understand the creation of image pyramids, splitting images into overlapping patches, and how these patches are processed through shared ViT encoders.

**Further Reading:**
- [Multi-Scale Vision Transformers](https://arxiv.org/abs/2104.11227)
- [DepthPro Paper Section 3.1: Network Architecture]

**Code Explanation:**

We'll implement code to create an image pyramid and split an image into overlapping patches, similar to how the `DepthProEncoder` processes images.

**Code:**

```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def create_image_pyramid(image, scales):
    """Create an image pyramid with specified scales."""
    pyramid = []
    for scale in scales:
        scaled_image = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
        pyramid.append(scaled_image)
    return pyramid

def split_into_patches(image, patch_size, overlap):
    """Split image into overlapping patches."""
    patches = []
    stride = int(patch_size * (1 - overlap))
    _, _, h, w = image.size()
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return torch.cat(patches, dim=0)

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((1536, 1536)),
    transforms.ToTensor(),
])
image = Image.open('sample_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Create image pyramid
scales = [1.0, 0.5, 0.25]
pyramid = create_image_pyramid(input_tensor, scales)

# Split the highest resolution image into overlapping patches
patch_size = 384
overlap_ratio = 0.25
patches = split_into_patches(pyramid[0], patch_size, overlap_ratio)

print("Number of patches:", patches.size(0))
print("Patch shape:", patches.shape)
```

**Notes:**

- We resize the input image to a fixed size of 1536x1536 pixels.
- The image pyramid consists of scaled versions of the input image.
- Overlapping patches are created from the highest resolution image to capture fine details.

**What to Read Next:**

- Study how overlapping patches improve feature representation: [Swin Transformer](https://arxiv.org/abs/2103.14030)
- Learn about the benefits of multi-scale features in depth estimation.

# Lesson 3: Working with the Decoder

**Concepts Covered:**
- Feature fusion in decoders
- Multiresolution convolutional decoders
- Implementing `MultiresConvDecoder`

**Overview:**

This lesson focuses on the decoder part of the DepthPro model. We'll look into how the `MultiresConvDecoder` combines features from multiple resolutions to produce a high-quality depth map.

**Further Reading:**
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [DepthPro Paper Section 3.1: Network Architecture]

**Code Explanation:**

We'll implement a simplified version of a multiresolution decoder that fuses features from different scales.

**Code:**

```python
import torch
import torch.nn as nn

class FeatureFusionBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.residual_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
        )
        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, x, res=None):
        if res is not None:
            x = self.skip_add.add(x, res)
        x = self.residual_conv(x)
        return x

class MultiresDecoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(num_features) for _ in range(4)
        ])
    
    def forward(self, features_list):
        x = features_list[-1]
        for i in reversed(range(len(features_list) - 1)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.fusion_blocks[i](x, features_list[i])
        return x

# Example feature maps from encoder at different resolutions
encoder_features = [
    torch.randn(1, 256, 192, 192),  # High-res feature
    torch.randn(1, 256, 96, 96),
    torch.randn(1, 256, 48, 48),
    torch.randn(1, 256, 24, 24),    # Low-res feature
]

decoder = MultiresDecoder(num_features=256)
decoded_features = decoder(encoder_features)

print("Decoded feature shape:", decoded_features.shape)
```

**Notes:**

- The `FeatureFusionBlock` enhances and combines features with residual connections.
- The `MultiresDecoder` upsamples features and fuses them progressively.
- We simulate encoder features with random tensors for demonstration.

**What to Read Next:**

- Explore residual networks: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Understand the importance of skip connections in deep learning architectures.

# Lesson 4: Building the DepthPro Model

**Concepts Covered:**
- Integrating the encoder and decoder
- Field of View (FoV) estimation with `FOVNetwork`
- Implementing the full DepthPro model

**Overview:**

In this lesson, we'll bring together the encoder and decoder to build the complete DepthPro model. We'll also discuss how the model estimates the field of view (FoV) using the `FOVNetwork` to produce metric depth maps without requiring camera intrinsics.

**Further Reading:**
- [DepthPro Paper Section 3.3: Focal Length Estimation]
- [Monocular Depth Estimation with Deep Learning: A Review](https://arxiv.org/abs/2003.06473)

**Code Explanation:**

We'll instantiate the DepthPro model, showing how the encoder and decoder are connected, and perform inference on an input image.

**Code:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming the encoder and decoder classes are defined as per previous lessons

class DepthPro(nn.Module):
    def __init__(self, encoder, decoder, fov_network):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fov_network = fov_network
        self.output_conv = nn.Conv2d(256, 1, kernel_size=1)
    
    def forward(self, x):
        encoder_features = self.encoder(x)
        features = self.decoder(encoder_features)
        depth = self.output_conv(features)
        fov = self.fov_network(x, features)
        return depth, fov

# Simplified encoder and decoder for demonstration
encoder = ...  # Use DepthProEncoder from previous lessons
decoder = ...  # Use MultiresConvDecoder as per Lesson 3
fov_network = ...  # Implement FOVNetwork if needed

model = DepthPro(encoder, decoder, fov_network)

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((1536, 1536)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
image = Image.open('sample_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    depth, fov = model(input_tensor)

print("Depth map shape:", depth.shape)
print("Estimated FoV:", fov)
```

**Notes:**

- The DepthPro model outputs both the depth map and estimated field of view.
- The FoV estimation allows the model to produce metric depth without known camera intrinsics.
- In practice, the encoder and decoder would be instances of `DepthProEncoder` and `MultiresConvDecoder` with learned weights.

**What to Read Next:**

- Understand how FoV estimation works in depth estimation models.
- Read about metric depth estimation and its challenges.

# Lesson 5: Evaluating Depth Maps and Boundary Metrics

**Concepts Covered:**
- Evaluation metrics for depth estimation
- Boundary accuracy in depth maps
- Implementing boundary metrics from `boundary_metrics.py`

**Overview:**

In the final lesson, we'll focus on evaluating the quality of depth maps, specifically the sharpness of boundaries. We'll learn about custom metrics introduced in DepthPro for assessing boundary accuracy, and implement code to compute these metrics on sample depth maps.

**Further Reading:**
- [DepthPro Paper Section 3.2: Evaluation Metrics for Sharp Boundaries]
- [Edge Detection and Image Segmentation](https://en.wikipedia.org/wiki/Edge_detection)

**Code Explanation:**

We'll use functions from `boundary_metrics.py` to compute boundary recall and F1 scores for predicted depth maps compared to ground truth.

**Code:**

```python
import numpy as np
from skimage import filters
from boundary_metrics import SI_boundary_F1, SI_boundary_Recall

# Assume we have predicted and ground truth depth maps
predicted_depth = depth.squeeze().cpu().numpy()
ground_truth_depth = ...  # Load your ground truth depth map as a NumPy array

# Compute scale-invariant boundary F1 score
f1_score = SI_boundary_F1(predicted_depth, ground_truth_depth)

print("Scale-invariant Boundary F1 Score:", f1_score)

# If ground truth is a binary mask (e.g., from segmentation)
ground_truth_mask = ...  # Load or compute a binary mask
recall = SI_boundary_Recall(predicted_depth, ground_truth_mask)

print("Scale-invariant Boundary Recall:", recall)
```

**Notes:**

- The `SI_boundary_F1` function evaluates the boundary accuracy between the predicted and ground truth depth maps.
- The metrics are scale-invariant, focusing on the correctness of depth discontinuities.
- Accurate boundary estimation is crucial for applications like novel view synthesis.

**What to Read Next:**

- Explore more on evaluation metrics in depth estimation research.
- Understand the importance of sharp boundaries in computer vision tasks.