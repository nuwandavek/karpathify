# Lesson 1: Introduction to Zero-Shot Metric Monocular Depth Estimation

*Understanding the basics of monocular depth estimation and the challenges in achieving zero-shot metric predictions.*

---

## Introduction

Monocular depth estimation involves predicting a depth map from a single RGB image. This depth map assigns a distance value to each pixel, indicating how far that part of the scene is from the camera. It's a fundamental task in computer vision with applications in 3D reconstruction, robotics, augmented reality, and more.

### Key Concepts:

- **Monocular Depth Estimation**: Estimating depth from a single image without stereo information.
- **Zero-Shot Learning**: The model can generalize to new, unseen data without additional training.
- **Metric Depth**: Depth estimates with absolute scale, meaning the distances correspond to real-world measurements.

### Challenges:

- **Ambiguity**: Inferring 3D structure from 2D images is inherently ambiguous.
- **Generalization**: Models often struggle to generalize to new domains or scenes.
- **Absolute Scale**: Estimating metric depth without camera parameters (like focal length) is challenging.

From the paper:

> "Our model, Depth Pro, produces metric depth maps with absolute scale on arbitrary images ‘in the wild’ without requiring metadata such as camera intrinsics."

---

## Objectives

- Understand the fundamentals of monocular depth estimation.
- Recognize the challenges in achieving zero-shot metric depth estimation.
- Prepare the environment for running code examples.

---

## Setup

First, let's set up the necessary environment. We'll be using Python and PyTorch for our implementations.

```python
# Install necessary libraries
# Uncomment the line below if running in a new environment
# !pip install torch torchvision matplotlib pillow requests

# Import libraries
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import requests
from io import BytesIO
```

---

## Loading and Displaying an Image

We'll work with a sample image throughout these lessons. Let's load and display it.

```python
# Load an example image
image_url = 'https://example.com/sample_image.jpg'  # Replace with a valid image URL

# Download the image
response = requests.get(image_url)
img = Image.open(BytesIO(response.content)).convert('RGB')

# Display the image
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title('Sample Image')
plt.axis('off')
plt.show()

# Optionally, save the image for later use
img.save('sample_image.jpg')
```

*Note: Replace `'https://example.com/sample_image.jpg'` with the URL of an actual image or use an image from your local machine.*

---

## Understanding Monocular Depth Estimation

In monocular depth estimation, our goal is to predict a depth map \( D \) from a single RGB image \( I \).

### Why is it challenging?

- **Lack of Stereo Information**: Without multiple viewpoints, depth cues are limited.
- **Scale Ambiguity**: Objects can appear differently based on perspective.
- **Generalization**: Scenes in the wild have diverse characteristics.

From the paper:

> "Zero-shot monocular depth estimation underpins a growing variety of applications... Our model, Depth Pro, produces metric depth maps with absolute scale on arbitrary images ‘in the wild’ without requiring metadata such as camera intrinsics."

---

## Conclusion

By the end of this lesson, we've established the foundational knowledge required to understand the problem Depth Pro aims to solve. In the next lesson, we'll delve into the architecture of Depth Pro and how it leverages Vision Transformers to overcome these challenges.

---

# Lesson 2: Depth Pro's Multi-Scale Vision Transformer Architecture

*Exploring how Depth Pro leverages a multi-scale Vision Transformer to produce high-resolution depth maps with sharp boundaries.*

---

## Overview

Depth Pro introduces an efficient multi-scale Vision Transformer (ViT) architecture to capture both global context and fine-grained details, enabling the production of sharp and high-resolution depth maps.

From the paper:

> "First, we design an efficient multi-scale ViT-based architecture for capturing the global image context while also adhering to fine structures at high resolution."

---

## Objectives

- Understand the basics of Vision Transformers (ViTs).
- Learn how Depth Pro modifies ViTs for multi-scale processing.
- Explore the encoder architecture in Depth Pro's implementation.

---

## Vision Transformers (ViTs)

ViTs apply the Transformer architecture, originally designed for natural language processing, to image data.

### Key Points:

- **Patch Embedding**: The image is divided into patches (e.g., 16x16 pixels), which are flattened and linearly projected to form patch embeddings.
- **Position Embedding**: Positional information is added to the patch embeddings.
- **Transformer Encoder**: The embeddings are processed through transformer layers to capture relationships between patches.

### Challenges with ViTs:

- **Computational Complexity**: The self-attention mechanism scales quadratically with the number of patches.
- **High-Resolution Images**: Directly applying ViTs to high-resolution images is computationally expensive.

---

## Depth Pro's Multi-Scale Approach

Depth Pro addresses the computational challenges by applying the ViT encoder to image patches at multiple scales and fusing the results.

From the paper:

> "We propose a network architecture that applies a plain ViT backbone at multiple scales and fuses predictions into a single high-resolution output."

### Steps in Depth Pro's Encoder:

1. **Create Image Pyramid**: Generate downsampled versions of the input image.
2. **Extract Patches at Each Scale**: Split each image in the pyramid into patches.
3. **Apply Shared ViT Encoder**: Process patches from all scales using the same ViT encoder.
4. **Merge Encodings**: Combine the outputs to form multi-resolution feature maps.
5. **Fuse Features**: Integrate features from different scales to capture both global context and fine details.

---

## Code Walkthrough

We'll examine parts of the encoder code to understand how it's implemented.

### Import Necessary Modules

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Encoder Class Initialization

```python
class DepthProEncoder(nn.Module):
    """DepthPro Encoder for multi-scale processing."""
    def __init__(self, dims_encoder, patch_encoder, image_encoder, hook_block_ids, decoder_features):
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder  # Shared ViT encoder
        self.image_encoder = image_encoder  # Global context encoder
        self.hook_block_ids = list(hook_block_ids)
        # Initialization of upsampling layers and hooks...
```

### Creating the Image Pyramid

```python
def _create_pyramid(self, x):
    x0 = x
    x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    x2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
    return x0, x1, x2
```

- **x0**: Original image (high resolution).
- **x1**: Downsampled by a factor of 0.5.
- **x2**: Downsampled by a factor of 0.25.

### Splitting Images into Patches

```python
def split(self, x, overlap_ratio=0.25):
    patch_size = 384  # For a standard ViT
    stride = int(patch_size * (1 - overlap_ratio))
    # Compute number of patches and extract them...
    # Returns concatenated patches
```

- **Overlap**: Ensures smooth transitions between patches.
- **Patches**: Extracted at different scales to capture varying levels of detail.

### Processing Patches with ViT Encoder

```python
def forward(self, x):
    x0, x1, x2 = self._create_pyramid(x)
    x0_patches = self.split(x0, overlap_ratio=0.25)
    x1_patches = self.split(x1, overlap_ratio=0.5)
    x2_patches = x2.unsqueeze(0)  # Single patch covering the whole image

    # Concatenate patches from all scales
    x_patches = torch.cat([x0_patches, x1_patches, x2_patches], dim=0)

    # Apply the shared ViT encoder
    x_encodings = self.patch_encoder(x_patches)

    # Merge encodings into feature maps
    # Fusion steps...
    return multi_scale_features
```

### Fusion of Features

- The outputs from different scales are combined.
- Feature maps are upsampled to a common resolution.
- Fused features capture both global context and fine details.

---

## Visualizing the Architecture

Here's a simplified representation:

![Depth Pro Encoder Architecture](https://i.imgur.com/your_image_link.png)

*Note: Replace with an actual diagram if available.*

---

## Practical Example

Since we don't have the actual ViT implementation here, we'll create mock encoders.

```python
class MockViTEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
    def forward(self, x):
        batch_size = x.size(0)
        return torch.rand(batch_size, self.output_dim)

# Instantiate mock encoders
patch_encoder = MockViTEncoder(output_dim=768)
image_encoder = MockViTEncoder(output_dim=768)

# Create the DepthPro encoder
encoder = DepthProEncoder(
    dims_encoder=[64, 128, 256, 512],
    patch_encoder=patch_encoder,
    image_encoder=image_encoder,
    hook_block_ids=[],
    decoder_features=256,
)

# Create a dummy input image
input_image = torch.randn(1, 3, 1536, 1536)

# Forward pass
features = encoder(input_image)

# Print the output feature shapes
for i, feat in enumerate(features):
    print(f"Feature {i} shape: {feat.shape}")
```

*Note: In the actual implementation, the ViT encoders process the patches and images to generate meaningful features.*

---

## Conclusion

In this lesson, we've explored how Depth Pro's multi-scale ViT architecture processes images at different resolutions to capture detailed features efficiently. In the next lesson, we'll delve into the training protocol and the loss functions used to train Depth Pro.

---

# Lesson 3: Training Protocol and Loss Functions

*Understanding how Depth Pro is trained to produce sharp, high-quality depth maps using a combination of real and synthetic datasets.*

---

## Overview

Depth Pro employs a carefully designed training curriculum that leverages both real-world and synthetic datasets. The goal is to achieve high generalization while also ensuring depth maps have sharp boundaries and fine details.

From the paper:

> "We devise a set of loss functions and a training curriculum that promote sharp depth estimates while training on real-world datasets that provide coarse and inaccurate supervision around boundaries, along with synthetic datasets that offer accurate pixelwise ground truth but limited realism."

---

## Objectives

- Learn about the two-stage training curriculum.
- Understand the loss functions used in training.
- Implement the loss functions in code.

---

## Two-Stage Training Curriculum

### Stage 1: Generalization

- **Goal**: Learn robust features that generalize across domains.
- **Datasets**: Mix of real-world and synthetic datasets.
- **Loss Functions**:
  - Mean Absolute Error (MAE) on metric datasets.
  - Scale-and-shift-invariant MAE on non-metric datasets.

### Stage 2: Boundary Sharpness

- **Goal**: Enhance sharpness and fine details in depth maps.
- **Datasets**: Only synthetic datasets with accurate pixel-wise ground truth.
- **Loss Functions**:
  - MAE, plus additional gradient-based losses.

---

## Loss Functions

### Canonical Inverse Depth

- **Definition**: Normalized depth representation prioritizing closer areas.

### Mean Absolute Error (MAE)

\[
\mathcal{L}_{\text{MAE}}(C, \hat{C}) = \frac{1}{N} \sum_{i=1}^N |C_i - \hat{C}_i|
\]

- \( C \): Predicted canonical inverse depth.
- \( \hat{C} \): Ground truth canonical inverse depth.
- \( N \): Number of valid pixels.

### Multi-Scale Gradient Losses

#### Mean Absolute Gradient Error (MAGE)

\[
\mathcal{L}_{\text{MAGE}} = \frac{1}{M} \sum_{j=1}^M \frac{1}{N_j} \sum_{i=1}^{N_j} \left| \nabla_S C_i^j - \nabla_S \hat{C}_i^j \right|
\]

#### Mean Absolute Laplace Error (MALE)

\[
\mathcal{L}_{\text{MALE}} = \frac{1}{M} \sum_{j=1}^M \frac{1}{N_j} \sum_{i=1}^{N_j} \left| \nabla_L C_i^j - \nabla_L \hat{C}_i^j \right|
\]

#### Mean Squared Gradient Error (MSGE)

\[
\mathcal{L}_{\text{MSGE}} = \frac{1}{M} \sum_{j=1}^M \frac{1}{N_j} \sum_{i=1}^{N_j} \left( \nabla_S C_i^j - \nabla_S \hat{C}_i^j \right)^2
\]

- \( M \): Number of scales.
- \( \nabla_S \): Scharr operator.
- \( \nabla_L \): Laplace operator.

---

## Implementing Loss Functions in Code

### Mean Absolute Error (MAE)

```python
def compute_mae_loss(pred, gt, mask=None):
    if mask is not None:
        valid = mask > 0
        loss = torch.abs(pred[valid] - gt[valid]).mean()
    else:
        loss = torch.abs(pred - gt).mean()
    return loss
```

### Gradient Operations

```python
def compute_gradient(img):
    # Scharr operator kernels
    scharr_x = torch.Tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]]).to(img.device) / 16
    scharr_y = torch.Tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]).to(img.device) / 16
    scharr_x = scharr_x.view(1, 1, 3, 3)
    scharr_y = scharr_y.view(1, 1, 3, 3)

    grad_x = F.conv2d(img, scharr_x, padding=1)
    grad_y = F.conv2d(img, scharr_y, padding=1)
    return grad_x, grad_y
```

### Mean Absolute Gradient Error (MAGE)

```python
def compute_mage_loss(pred, gt):
    pred_grad_x, pred_grad_y = compute_gradient(pred)
    gt_grad_x, gt_grad_y = compute_gradient(gt)
    loss = (torch.abs(pred_grad_x - gt_grad_x) + torch.abs(pred_grad_y - gt_grad_y)).mean()
    return loss
```

---

## Example Training Loop

```python
# Assume we have a model, optimizer, and dataloaders
for epoch in range(num_epochs):
    for data in dataloader:
        images, gt_depth = data  # Load your data batch
        optimizer.zero_grad()
        pred_depth = model(images)
        loss_mae = compute_mae_loss(pred_depth, gt_depth)
        loss_mage = compute_mage_loss(pred_depth, gt_depth)
        loss = loss_mae + loss_mage  # Combine losses as needed
        loss.backward()
        optimizer.step()
```

*Note: This is a simplified example. In practice, you'll need to handle data loading, masking invalid pixels, multi-scale losses, etc.*

---

## Discussion

By combining the MAE loss with gradient-based losses, Depth Pro encourages the model to produce depth maps that are not only accurate in terms of depth values but also have sharp transitions and details.

---

## Conclusion

In this lesson, we've learned about the training strategy and loss functions that enable Depth Pro to produce high-quality, sharp depth maps. In the next lesson, we'll explore the evaluation metrics used to assess boundary accuracy and how they're implemented.

---

# Lesson 4: Boundary Accuracy Metrics and Evaluation

*Understanding the importance of sharp boundaries in depth estimation and how Depth Pro evaluates and improves boundary accuracy.*

---

## Overview

Sharp boundaries in depth maps are crucial for applications like novel view synthesis, where blurry or misaligned edges can lead to visual artifacts. Depth Pro introduces dedicated evaluation metrics for boundary accuracy.

From the paper:

> "We derive a new set of metrics that enable leveraging highly accurate matting datasets for quantifying the accuracy of boundary tracing in evaluating monocular depth maps."

---

## Objectives

- Learn why boundary accuracy matters in depth estimation.
- Understand the boundary evaluation metrics introduced in the paper.
- Implement the metrics in code.

---

## Importance of Sharp Boundaries

- **Visual Quality**: Blurry edges in depth maps can result in "flying pixels" or artifacts in rendered images.
- **Applications**: Tasks like image segmentation, matting, and augmented reality require precise depth discontinuities.

---

## Boundary Evaluation Metrics

### Pairwise Depth Ratio

To determine if there's an occluding contour between neighboring pixels \( i \) and \( j \), we define:

\[
c_d(i, j) = \left[ \frac{d(j)}{d(i)} > 1 + \frac{t}{100} \right]
\]

Where:
- \( d(i) \): Depth at pixel \( i \).
- \( t \): Threshold percentage.

### Precision and Recall

- **Precision (\( P \))**: The proportion of predicted boundaries that are correct.
- **Recall (\( R \))**: The proportion of true boundaries that are detected.

\[
P(t) = \frac{\sum_{i,j} c_d(i,j) \wedge c_{\hat{d}}(i,j)}{\sum_{i,j} c_d(i,j)} \\
R(t) = \frac{\sum_{i,j} c_d(i,j) \wedge c_{\hat{d}}(i,j)}{\sum_{i,j} c_{\hat{d}}(i,j)}
\]

### Boundary F1 Score

The harmonic mean of precision and recall:

\[
F1 = 2 \times \frac{P \times R}{P + R}
\]

---

## Implementing Boundary Metrics in Code

### Defining Occluding Contours

```python
def compute_contours(depth_map, threshold):
    # Compute depth ratios between neighboring pixels
    depth_ratio_right = depth_map[:, :, 1:] / depth_map[:, :, :-1]
    depth_ratio_down = depth_map[:, 1:, :] / depth_map[:, :-1, :]

    # Determine occluding contours based on the threshold
    contour_right = depth_ratio_right > (1 + threshold / 100)
    contour_down = depth_ratio_down > (1 + threshold / 100)
    return contour_right, contour_down
```

### Precision and Recall Calculation

```python
def compute_precision_recall(pred_contour, gt_contour):
    true_positive = (pred_contour & gt_contour).sum()
    pred_positive = pred_contour.sum()
    gt_positive = gt_contour.sum()

    precision = true_positive / (pred_positive + 1e-6)
    recall = true_positive / (gt_positive + 1e-6)
    return precision.item(), recall.item()
```

### Boundary F1 Score

```python
def compute_boundary_f1(pred_depth, gt_depth, thresholds):
    f1_scores = []
    for t in thresholds:
        pred_contours = compute_contours(pred_depth, t)
        gt_contours = compute_contours(gt_depth, t)
        precision, recall = compute_precision_recall(pred_contours, gt_contours)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)
```

*Note: Ensure that depth maps are properly normalized and handle cases where division by zero might occur.*

---

## Example Evaluation

```python
# Assume pred_depth and gt_depth are numpy arrays of the same shape
thresholds = np.linspace(5, 25, 5)  # Thresholds from 5% to 25%
boundary_f1 = compute_boundary_f1(pred_depth, gt_depth, thresholds)
print(f'Boundary F1 Score: {boundary_f1}')
```

---

## Discussion

By evaluating the model based on boundary accuracy, Depth Pro ensures that it's not just the overall depth values that are accurate, but also the transitions and edges, which are critical for high-quality rendering.

---

## Conclusion

In this lesson, we've delved into how Depth Pro measures and improves boundary accuracy in depth estimation. In the final lesson, we'll explore how Depth Pro estimates focal length and how to perform inference using the trained model.

---

# Lesson 5: Focal Length Estimation and Inference with Depth Pro

*Understanding how Depth Pro estimates focal length from a single image and how to use the model for inference.*

---

## Overview

Depth Pro can estimate the focal length (field of view) from a single image, enabling metric depth estimation without requiring camera intrinsics.

From the paper:

> "We contribute zero-shot focal length estimation from a single image that dramatically outperforms the prior state of the art."

---

## Objectives

- Learn how Depth Pro estimates focal length.
- Run inference using Depth Pro on sample images.
- Understand how to obtain metric depth maps.

---

## Focal Length Estimation

### Why Estimate Focal Length?

- **Metric Depth**: To convert relative depth estimates to metric measurements, focal length is needed.
- **Camera Metadata**: EXIF data may be missing or inaccurate, especially for images "in the wild."

### Approach

- **Separate Focal Length Estimation Head**: A small network predicts the horizontal field of view (FoV) based on features from the depth estimation network.
- **Training**: The focal length head is trained separately using images with known focal lengths.

---

## Code Implementation

### Focal Length Network

```python
class FOVNetwork(nn.Module):
    """Field of View estimation network."""
    def __init__(self, num_features):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(num_features // 2, num_features // 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(num_features // 4, 1, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):
        return self.head(x)
```

### Integrating with Depth Pro

```python
class DepthPro(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fov_network = FOVNetwork(num_features=encoder.output_dim)

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        fov = self.fov_network(features)
        return depth, fov
```

---

## Inference with Depth Pro

### Loading a Pre-trained Model

```python
# Assume model weights are saved in 'depth_pro.pth'
model = DepthPro(encoder, decoder)
model.load_state_dict(torch.load('depth_pro.pth'))
model.eval()
```

### Running Inference on an Image

```python
# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((1536, 1536)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
img = Image.open('sample_image.jpg').convert('RGB')
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Run the model
with torch.no_grad():
    depth, fov = model(input_tensor)

# Convert depth to numpy array
depth_map = depth.squeeze().cpu().numpy()
fov_value = fov.item()
```

### Converting Depth to Metric Units

- **Canonical Inverse Depth**: The model outputs canonical inverse depth.
- **Conversion to Metric Depth**:

\[
D_{\text{metric}} = \frac{f_{\text{pixels}}}{w} \times C
\]

- \( f_{\text{pixels}} \): Focal length in pixels.
- \( w \): Image width in pixels.
- \( C \): Canonical inverse depth.

- **Calculating Focal Length in Pixels**:

\[
f_{\text{pixels}} = \frac{0.5 \times w}{\tan(0.5 \times \text{FoV (in radians)})}
\]

---

## Visualizing the Results

```python
import matplotlib.pyplot as plt

# Display the depth map
plt.imshow(depth_map, cmap='plasma')
plt.colorbar(label='Depth (arbitrary units)')
plt.title('Predicted Depth Map')
plt.axis('off')
plt.show()
```

---

## Discussion

Depth Pro's ability to estimate focal length allows it to produce metric depth maps without relying on camera metadata. This is particularly useful for images where such data is unavailable or unreliable.

---

## Conclusion

In this final lesson, we've learned how Depth Pro estimates focal length and performs inference to generate metric depth maps from single images. With this knowledge, you're now equipped to understand and work with Depth Pro's implementation and contribute to advancements in monocular depth estimation.

---

# Final Remarks

Throughout these lessons, we've explored Depth Pro's approach to zero-shot metric monocular depth estimation. By understanding the architecture, training protocol, evaluation metrics, and inference process, you can now delve deeper into the codebase and potentially contribute to this exciting area of research.

---