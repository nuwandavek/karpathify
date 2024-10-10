# Lesson 1: Introduction to Monocular Depth Estimation and Depth Pro

In this lesson, we will introduce the concept of **monocular depth estimation**, understand the challenges associated with it, and get an overview of the **Depth Pro** model. Finally, we'll run a simple inference using Depth Pro on a sample image.

---

## What is Monocular Depth Estimation?

**Monocular depth estimation** is the task of predicting the depth of each pixel in an image captured by a single camera. Unlike stereo vision, which relies on multiple images from different viewpoints, monocular depth estimation aims to infer depth information using only one image.

### Challenges:

- **Ambiguity**: A single image lacks direct 3D information, making depth estimation inherently ambiguous.
- **Scale Recovery**: Without knowledge of the camera parameters (like focal length), it's difficult to recover the absolute scale of objects in the scene.
- **Generalization**: Models trained on specific datasets may not perform well on unseen images due to varying scene types and camera settings.

---

## Introducing Depth Pro

**Depth Pro** is a state-of-the-art model for **zero-shot metric monocular depth estimation**. It addresses the challenges mentioned above by:

- Producing **sharp and detailed** depth maps with high-frequency details.
- Estimating **metric depth** with **absolute scale** without relying on camera intrinsics.
- Being **fast and efficient**, capable of generating high-resolution depth maps quickly.

### Key Features:

- **Efficient Multi-scale Vision Transformer (ViT) Architecture**: Utilizes ViTs for dense prediction tasks.
- **Training Protocol with Real and Synthetic Data**: Combines datasets to improve generalization and accuracy.
- **Boundary Accuracy Metrics**: Introduces new metrics to evaluate and improve boundary sharpness.
- **Focal Length Estimation**: Predicts camera focal length from a single image to achieve metric scale.

---

## Running an Inference with Depth Pro

In this section, we will:

- Install the required libraries.
- Load the pre-trained Depth Pro model.
- Run inference on a sample image.
- Visualize the estimated depth map.

---

### Step 1: Install Required Libraries

```python
# Install PyTorch and other dependencies if not already installed
# Uncomment and run the following line if needed:

# !pip install torch torchvision pillow matplotlib
```

*Note:* Ensure that you have PyTorch installed in your environment. The above command installs PyTorch and other required libraries.

---

### Step 2: Import Necessary Modules

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
```

---

### Step 3: Load the Depth Pro Model

We'll use the `create_model_and_transforms` function from the `depth_pro` module (provided in the repository) to load the pre-trained model and the necessary data transforms.

First, ensure that the repository code is accessible. You can clone it or copy the `depth_pro` module into your working directory.

```python
# Assuming you have the 'depth_pro' module in your working directory
from depth_pro import create_model_and_transforms

# Determine the device to use (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and the data transforms
model, transform = create_model_and_transforms(device=device)
model.eval()  # Set the model to evaluation mode
```

**Explanation:**

- We import `create_model_and_transforms` from the `depth_pro` module.
- We check if a GPU is available and set the device accordingly.
- We call `create_model_and_transforms` to load the pre-trained Depth Pro model and the necessary data transforms.
- We set the model to evaluation mode using `model.eval()` since we're not training it.

---

### Step 4: Load a Sample Image

Select an image to test the model. You can use any RGB image.

```python
# Path to the sample image
image_path = 'sample_image.jpg'  # Replace with your image path

# Load the image using PIL
image = Image.open(image_path).convert('RGB')

# Display the image
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis('off')
plt.title('Input Image')
plt.show()
```

**Explanation:**

- We specify the path to the image and load it using `Image.open`.
- We convert the image to RGB to ensure it has three channels.
- We display the image using `matplotlib`.

---

### Step 5: Preprocess the Image and Run Inference

```python
# Apply the necessary transforms to the image
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Move the tensor to the appropriate device
input_tensor = input_tensor.to(device)

# Run inference
with torch.no_grad():
    prediction = model.infer(input_tensor)

# Extract the depth map from the prediction
depth_map = prediction['depth'].cpu().numpy().squeeze()
```

**Explanation:**

- We apply the transforms to the image to prepare it for the model.
    - The transforms include resizing, normalization, and conversion to a tensor.
- We add a batch dimension since the model expects a batched input.
- We move the input tensor to the same device as the model.
- We use `torch.no_grad()` to prevent PyTorch from tracking gradients during inference.
- We call the `infer` method of the model to get the prediction dictionary.
- We extract the depth map from the prediction and convert it to a NumPy array for visualization.

---

### Step 6: Visualize the Estimated Depth Map

```python
# Normalize the depth map for visualization
depth_min = depth_map.min()
depth_max = depth_map.max()
depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)

# Display the depth map
plt.figure(figsize=(8, 6))
plt.imshow(depth_normalized, cmap='viridis')
plt.axis('off')
plt.title('Estimated Depth Map')
plt.colorbar()
plt.show()
```

**Explanation:**

- Depth values can vary widely, so we normalize them to the range [0, 1] for visualization.
- We use the `viridis` colormap for better visual contrast.
- We display the normalized depth map using `matplotlib` and add a colorbar to show the scale.

---

## Summary

In this lesson, we:

- Introduced the concept of monocular depth estimation and its challenges.
- Learned about Depth Pro and its key features.
- Ran an inference using Depth Pro on a sample image.
- Visualized the estimated depth map.

Now that we've seen Depth Pro in action, in the next lesson, we'll delve into the basics of Vision Transformers (ViT) and how they're applied in Depth Pro.

---

# End of Lesson 1

---

# Lesson 2: Vision Transformers (ViT) in Depth Pro

In this lesson, we'll introduce **Vision Transformers (ViT)**, understand their key concepts, and explore how Depth Pro utilizes ViTs in its architecture.

---

## Understanding Vision Transformers (ViT)

### Background

- Originally, **Transformers** were introduced for sequence modeling tasks in NLP (e.g., machine translation).
- **ViTs** adapt the Transformer architecture for image data.

### Key Concepts

1. **Patch Embedding**:
   - The input image is divided into fixed-size patches (e.g., 16x16 pixels).
   - Each patch is flattened and mapped to a vector (embedding) of a fixed size.

2. **Positional Encoding**:
   - Since Transformers lack inherent positional information, positional encodings are added to the patch embeddings to retain spatial information.

3. **Self-Attention Mechanism**:
   - Allows the model to weigh the importance of different patches when making predictions.
   - Captures global dependencies in the image.

### Benefits for Computer Vision

- **Global Context**: ViTs can capture relationships across the entire image.
- **Scalability**: Can be scaled to larger datasets and models effectively.
- **Flexibility**: Can be applied to various vision tasks, including classification, segmentation, and depth estimation.

---

## ViT in Depth Pro

Depth Pro leverages ViTs for both the **patch encoder** and the **image encoder**. The model processes the image at multiple scales to capture both local details and global context.

### Multi-scale Approach

- **Patch Encoder**:
  - Processes overlapping patches at different scales.
  - Shared weights across scales to enforce scale invariance.

- **Image Encoder**:
  - Processes the downsampled whole image to provide global context.

### Code Exploration

Let's explore how ViTs are integrated into Depth Pro by examining key code snippets.

---

### Step 1: Creating the ViT Backbones

```python
# File: depth_pro.py

def create_backbone_model(preset: ViTPreset) -> Tuple[nn.Module, ViTPreset]:
    ...
    if preset in VIT_CONFIG_DICT:
        config = VIT_CONFIG_DICT[preset]
        model = create_vit(preset=preset, use_pretrained=False)
    else:
        raise KeyError(f"Preset {preset} not found.")
    ...
    return model, config
```

**Explanation:**

- **Presets**: Pre-defined configurations for different ViT models are stored in `VIT_CONFIG_DICT`.
- **create_vit**: A function that creates a ViT model based on the preset.
- **Backbone Models**: Both the patch encoder and image encoder are created using this function.

---

### Step 2: Understanding `create_vit` Function

```python
# File: network/vit_factory.py

def create_vit(...):
    ...
    model = timm.create_model(
        config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
    )
    model = make_vit_b16_backbone(
        model,
        encoder_feature_dims=config.encoder_feature_dims,
        encoder_feature_layer_ids=config.encoder_feature_layer_ids,
        vit_features=config.embed_dim,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    ...
    return model.model
```

**Explanation:**

- **timm Library**: Third-party library `timm` is used to create the ViT model.
- **Dynamic Image Size**: Allows the model to adjust to different input sizes.
- **make_vit_b16_backbone**: Wraps the model to adapt it for Depth Pro's requirements, including extracting features from specific layers.

---

### Step 3: ViT in Depth Pro Encoder

```python
# File: network/encoder.py

class DepthProEncoder(nn.Module):
    def __init__(...):
        ...
        # Patch and image encoders
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        ...
```

**Explanation:**

- **Patch Encoder**: Processes image patches at multiple scales.
- **Image Encoder**: Processes the downsampled full image.
- **Shared Weights**: The same ViT architecture (and possibly weights) can be used for both encoders.

---

### Step 4: Extracting Features

```python
def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    ...
    # Split the image into patches at different scales
    x_pyramid_patches = torch.cat(
        (x0_patches, x1_patches, x2_patches), dim=0
    )
    # Encode the patches using the ViT patch encoder
    x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
    ...
```

**Explanation:**

- **Multi-scale Patches**: The image is split into patches with overlapping regions to capture fine details.
- **Encoding**: The ViT model processes these patches to produce feature representations.

---

### Step 5: ViT Architecture Details

While the low-level implementation details are abstracted away, it's important to understand that the ViT model consists of:

- **Multi-Head Self-Attention Layers**: Capture relationships between patches.
- **Feedforward Networks**: Apply non-linear transformations to the features.
- **Layer Normalization and Residual Connections**: Help with training stability and model performance.

---

### Visualizing the ViT Process

Here's a simplified diagram of how ViT processes an image:

1. **Input Image**: The image is divided into patches.
2. **Patch Embedding**: Each patch is flattened and passed through a linear layer.
3. **Positional Encoding**: Added to the patch embeddings to retain spatial information.
4. **Transformer Encoder**: Processes the sequence of embeddings through self-attention layers.
5. **Output Features**: The encoded features represent the input image.

---

## Summary

In this lesson, we:

- Introduced the concept of Vision Transformers and their key components.
- Explored how ViTs are integrated into Depth Pro as patch and image encoders.
- Reviewed code snippets to understand how the ViTs are created and used in the model.

Understanding ViTs is crucial for grasping how Depth Pro effectively captures both local and global information in images for depth estimation.

In the next lesson, we'll delve deeper into Depth Pro's architecture, focusing on how it processes multi-scale features and fuses them to produce accurate depth maps.

---

# End of Lesson 2

---

# Lesson 3: Depth Pro's Multi-scale Architecture

In this lesson, we'll explore Depth Pro's multi-scale architecture in detail, focusing on the **encoder**, **decoder**, and how the model fuses features from different scales to produce high-resolution depth maps.

---

## Overview of Depth Pro's Architecture

![Depth Pro Architecture](https://raw.githubusercontent.com/your-repo/your-image-path/architecture.png)

*Note: Replace the image link with the actual architecture image from the paper or repository.*

### Key Components:

1. **Input Image Pyramid**: The input image is downsampled to create an image pyramid with multiple scales.
2. **Patch Encoder**: Processes patches from different scales using a shared ViT encoder.
3. **Image Encoder**: Processes the downsampled full image to provide global context.
4. **Feature Fusion**: Combines features from different scales in the decoder.
5. **Decoder**: Generates the high-resolution depth map from fused features.

---

## Multi-scale Encoding

### Creating the Image Pyramid

```python
# File: network/encoder.py

def _create_pyramid(self, x: torch.Tensor) -> tuple:
    # Original resolution (e.g., 1536x1536)
    x0 = x

    # Downsampled by a factor of 2 (e.g., 768x768)
    x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

    # Downsampled by a factor of 4 (e.g., 384x384)
    x2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

    return x0, x1, x2
```

**Explanation:**

- **x0**: Original high-resolution image.
- **x1**: Medium-resolution image.
- **x2**: Low-resolution image.
- The pyramid allows the model to capture features at different scales.

### Splitting Images into Patches

```python
def split(self, x: torch.Tensor, overlap_ratio: float) -> torch.Tensor:
    patch_size = 384  # Size of patches
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = x.shape[-1]  # Assuming square images

    steps = (image_size - patch_size) // patch_stride + 1
    patches = []

    for j in range(steps):
        for i in range(steps):
            x0 = i * patch_stride
            y0 = j * patch_stride
            x1 = x0 + patch_size
            y1 = y0 + patch_size
            patch = x[:, :, y0:y1, x0:x1]
            patches.append(patch)

    return torch.cat(patches, dim=0)
```

**Explanation:**

- **Overlap**: Controlled by `overlap_ratio` to capture continuous features.
- **Patches**: Extracted from the image and concatenated to form a batch.

### Encoding Patches with ViT Patch Encoder

```python
# Encode patches
x_pyramid_patches = torch.cat([x0_patches, x1_patches, x2_patches], dim=0)
x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
```

**Explanation:**

- Patches from different scales are concatenated.
- The shared `patch_encoder` processes all patches.

---

## Feature Fusion and Decoding

### Merging Encoded Patches

```python
def merge(self, x: torch.Tensor, batch_size: int, padding: int) -> torch.Tensor:
    # Calculate the number of steps (patches) along each dimension
    steps = int(np.sqrt(x.shape[0] // batch_size))
    # ...
    # Reconstruct the feature map from the patches
    # ...
    return merged_feature_map
```

**Explanation:**

- **Purpose**: Reconstruct the spatial arrangement of features.
- **Padding**: Removed to ensure seamless merging.

### Decoder Architecture

```python
# File: network/decoder.py

class MultiresConvDecoder(nn.Module):
    def __init__(self, dims_encoder: Iterable[int], dim_decoder: int):
        ...
        # Convolutional layers to adjust dimensions
        self.convs = nn.ModuleList([
            nn.Conv2d(dim_in, dim_decoder, kernel_size=1, bias=False)
            for dim_in in dims_encoder
        ])
        # Fusion blocks
        self.fusions = nn.ModuleList([
            FeatureFusionBlock2d(dim_decoder) for _ in dims_encoder
        ])

    def forward(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        features = encodings[-1]
        for i in reversed(range(len(encodings) - 1)):
            features = self.fusions[i](features, self.convs[i](encodings[i]))
        return features
```

**Explanation:**

- **Convs**: Adjust the number of channels in the encoded features.
- **Fusions**: Merge features from different scales.

### Feature Fusion Block

```python
class FeatureFusionBlock2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor = None) -> torch.Tensor:
        if skip_connection is not None:
            x = x + skip_connection
        x = self.residual_block(x) + x
        return x
```

**Explanation:**

- **Skip Connections**: Adds features from previous layers (like in ResNet).
- **Residual Block**: Helps in learning identity mappings, improving gradient flow.

---

## Final Depth Prediction

```python
# File: depth_pro.py

class DepthPro(nn.Module):
    def __init__(...):
        ...
        self.head = nn.Sequential(
            nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim_decoder // 2, dim_decoder // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_decoder // 2, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
        features = self.decoder(encodings)
        depth = self.head(features)
        return depth
```

**Explanation:**

- **Head**: Transforms the fused features into the final depth map.
- **ConvTranspose2d**: Upsamples the feature map to the desired resolution.

---

## Summary

In this lesson, we:

- Explored Depth Pro's multi-scale architecture.
- Learned how the model processes images at different scales and patches.
- Understood how features are fused in the decoder to produce the depth map.
- Reviewed code snippets to see how these components are implemented.

By leveraging multi-scale processing and effective feature fusion, Depth Pro achieves high-resolution depth estimation with sharp details.

In the next lesson, we'll delve into how Depth Pro is trained, including the loss functions used and the training protocol.

---

# End of Lesson 3

---

# Lesson 4: Training Depth Pro - Loss Functions and Training Protocol

In this lesson, we'll discuss how Depth Pro is trained, focusing on the **loss functions** and the **training curriculum** that combines real and synthetic datasets to achieve high accuracy and sharp boundaries.

---

## Loss Functions Used in Depth Pro

To train Depth Pro effectively, several loss functions are employed to handle different aspects of depth estimation.

### 1. Mean Absolute Error (MAE)

**Definition:**

\[
\mathcal{L}_{\text{MAE}}(\hat{C}, C) = \frac{1}{N} \sum_{i=1}^{N} |\hat{C}_i - C_i|
\]

- \( \hat{C} \): Predicted canonical inverse depth.
- \( C \): Ground truth canonical inverse depth.
- \( N \): Number of valid pixels.

**Purpose:**

- Penalizes the absolute differences between predicted and ground truth depths.
- Robust to outliers.

### 2. Multi-scale Gradient Losses

To encourage sharpness and preserve edges, gradient-based losses are applied at multiple scales.

**Definition:**

\[
\mathcal{L}_{*, p, M}(C, \hat{C}) = \frac{1}{M} \sum_{j=1}^{M} \frac{1}{N_j} \sum_{i=1}^{N_j} |\nabla_* C_i^j - \nabla_* \hat{C}_i^j|^p
\]

- \( \nabla_* \): Spatial derivative operator (e.g., gradient, Laplacian).
- \( p \): Norm (usually 1 for MAE or 2 for MSE).
- \( M \): Number of scales.
- \( j \): Scale index.

**Purpose:**

- Encourages the model to focus on edges and fine details.
- Improves boundary sharpness in the predicted depth maps.

---

## Training Curriculum

### Stage 1: Generalization

- **Datasets**: A mix of real-world and synthetic datasets.
- **Objective**: Learn robust features that generalize across different domains.
- **Losses**:
  - **MAE Loss**: Applied to metric datasets.
  - **Scale-and-Shift-Invariant Loss**: Applied to non-metric datasets (datasets without absolute scale).

### Stage 2: Refinement

- **Datasets**: Only high-quality synthetic datasets.
- **Objective**: Enhance the sharpness and accuracy of depth boundaries.
- **Losses**:
  - **MAE Loss**: Enforces overall depth accuracy.
  - **Multi-scale Gradient Losses**: Focuses on preserving edges and details.

**Rationale:**

- **Stage 1**: Real-world datasets provide diversity, while synthetic datasets offer precise ground truth.
- **Stage 2**: Synthetic datasets help refine the model to capture fine details, leveraging their accurate annotations.

---

## Code Exploration: Loss Function Implementation

While the training code isn't provided in the repository snippet, we can discuss how the loss functions might be implemented in PyTorch.

### Implementing MAE Loss

```python
import torch.nn.functional as F

def compute_mae_loss(predicted_depth, ground_truth):
    # Assume both inputs are of shape [Batch, 1, Height, Width]
    loss = F.l1_loss(predicted_depth, ground_truth, reduction='mean')
    return loss
```

---

### Computing Gradients for Gradient Loss

```python
def gradient(image):
    # Computes image gradients using Sobel operator
    gradient_x = image[:, :, :, :-1] - image[:, :, :, 1:]
    gradient_y = image[:, :, :-1, :] - image[:, :, 1:, :]
    return gradient_x, gradient_y
```

### Implementing Multi-scale Gradient Loss

```python
def multi_scale_gradient_loss(predicted_depth, ground_truth, scales=[1, 0.5, 0.25]):
    total_loss = 0
    for scale in scales:
        if scale != 1:
            # Downsample the predicted and ground truth depths
            predicted_scaled = F.interpolate(predicted_depth, scale_factor=scale, mode='bilinear', align_corners=False)
            ground_truth_scaled = F.interpolate(ground_truth, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            predicted_scaled = predicted_depth
            ground_truth_scaled = ground_truth
        
        # Compute gradients
        pred_grad_x, pred_grad_y = gradient(predicted_scaled)
        gt_grad_x, gt_grad_y = gradient(ground_truth_scaled)
        
        # Compute gradient loss
        loss_x = F.l1_loss(pred_grad_x, gt_grad_x, reduction='mean')
        loss_y = F.l1_loss(pred_grad_y, gt_grad_y, reduction='mean')
        
        total_loss += (loss_x + loss_y)
    
    return total_loss / len(scales)
```

---

### Total Loss Function

Combining both MAE loss and multi-scale gradient loss:

```python
def total_loss(predicted_depth, ground_truth):
    mae = compute_mae_loss(predicted_depth, ground_truth)
    grad_loss = multi_scale_gradient_loss(predicted_depth, ground_truth)
    return mae + grad_loss_weight * grad_loss  # grad_loss_weight is a hyperparameter
```

---

### Training Loop Outline

```python
for batch in dataloader:
    images, depths = batch
    images = images.to(device)
    depths = depths.to(device)

    optimizer.zero_grad()
    
    # Forward pass
    predicted_depth = model(images)
    
    # Compute loss
    loss = total_loss(predicted_depth, depths)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
```

---

## Summary

In this lesson, we:

- Discussed the loss functions used to train Depth Pro, focusing on MAE and multi-scale gradient losses.
- Understood the two-stage training curriculum that combines real and synthetic data.
- Explored how these loss functions encourage both accuracy and sharpness in the depth maps.

In the next lesson, we'll look at how Depth Pro evaluates boundary accuracy and how it estimates focal length from a single image.

---

# End of Lesson 4

---

# Lesson 5: Evaluating Boundary Accuracy and Focal Length Estimation

In this final lesson, we'll explore how Depth Pro evaluates boundary accuracy using new metrics and how it estimates focal length from a single image to produce metric depth maps without relying on camera intrinsics.

---

## Evaluating Boundary Accuracy

### Importance of Boundary Sharpness

- **Visual Quality**: Sharp boundaries in depth maps lead to better visualizations and applications like 3D rendering.
- **Downstream Tasks**: Accurate boundaries improve the performance of tasks like segmentation and object recognition.

### New Metrics Introduced

Depth Pro introduces metrics that leverage existing datasets with accurate masks (e.g., matting datasets) to evaluate boundary accuracy.

---

### Occluding Contours in Depth Maps

An **occluding contour** between two neighboring pixels \( i \) and \( j \) is identified if the depth difference exceeds a threshold:

\[
c_d(i, j) = \left[ \frac{d(j)}{d(i)} > (1 + \frac{t}{100}) \right]
\]

- \( d(i) \): Depth at pixel \( i \).
- \( t \): Threshold percentage (e.g., 5%).

### Precision and Recall Calculations

- **Precision** (\( P \)) and **Recall** (\( R \)) are calculated based on the predicted and ground truth occluding contours.

\[
P(t) = \frac{\sum_{i, j} c_d(i, j) \land c_{\hat{d}}(i, j)}{\sum_{i, j} c_d(i, j)}
\]

\[
R(t) = \frac{\sum_{i, j} c_d(i, j) \land c_{\hat{d}}(i, j)}{\sum_{i, j} c_{\hat{d}}(i, j)}
\]

- \( c_d \): Ground truth occluding contours.
- \( c_{\hat{d}} \): Predicted occluding contours.
- \( \land \): Logical AND operation.

### F1 Score

- Combines precision and recall to provide a balanced metric:

\[
F1 = 2 \times \frac{P \times R}{P + R}
\]

---

## Code Exploration: Boundary Metrics

```python
# File: eval/boundary_metrics.py

def boundary_f1(pr: np.ndarray, gt: np.ndarray, t: float) -> float:
    # Compute occluding contours for predicted and ground truth depths
    ap, bp, cp, dp = fgbg_depth(pr, t)
    ag, bg, cg, dg = fgbg_depth(gt, t)

    # Calculate recall
    r = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1) +
        np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1) +
        np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1) +
        np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )

    # Calculate precision
    p = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ap), 1) +
        np.count_nonzero(bp & bg) / max(np.count_nonzero(bp), 1) +
        np.count_nonzero(cp & cg) / max(np.count_nonzero(cp), 1) +
        np.count_nonzero(dp & dg) / max(np.count_nonzero(dp), 1)
    )

    # Compute F1 score
    if r + p == 0:
        return 0.0
    return 2 * (r * p) / (r + p)
```

**Explanation:**

- **fgbg_depth**: Computes foreground/background relationships between neighboring pixels.
- **Directional Calculations**: Evaluates in four directions (left, top, right, bottom).
- **Normalization**: Ensures division by zero is avoided using `max(count, 1)`.

---

## Focal Length Estimation from a Single Image

### Motivation

- **Metric Depth**: To recover absolute depth values (metric), the camera's focal length is usually required.
- **Applicability**: Estimating the focal length directly from the image allows the model to work without camera intrinsics.

### Implementation

```python
# File: network/fov.py

class FOVNetwork(nn.Module):
    def __init__(self, num_features: int, fov_encoder: Optional[nn.Module] = None):
        super().__init__()
        # Simplified for illustration
        self.head = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features // 2, 1)
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # Use features from the encoder
        fov = self.head(features)
        return fov
```

**Explanation:**

- **Input Features**: Uses features from the encoder or decoder.
- **Prediction**: Outputs a single scalar representing the estimated focal length or field of view.

### Incorporating FOV Estimation into Depth Prediction

```python
# File: depth_pro.py

class DepthPro(nn.Module):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...
        # Get depth predictions
        depth = self.head(features)
        # Estimate focal length
        fov_deg = self.fov(x, features)
        # Adjust depth to metric using estimated focal length
        if fov_deg is not None:
            f_px = compute_focal_length_in_pixels(fov_deg, image_width)
            depth = adjust_depth_to_metric(depth, f_px)
        return depth, fov_deg
```

---

## Summary

In this lesson, we:

- Learned about the new metrics introduced in Depth Pro to evaluate boundary accuracy.
- Understood how these metrics focus on occluding contours and edges in depth maps.
- Explored how Depth Pro estimates the focal length from a single image, eliminating the need for camera intrinsics.

---

# End of Lesson 5

---

Congratulations! You've completed all five lessons on Depth Pro. You've gained an understanding of:

- Monocular depth estimation challenges and Depth Pro's contributions.
- The use of Vision Transformers in Depth Pro's architecture.
- How Depth Pro processes multi-scale features and fuses them for depth prediction.
- The training process, loss functions, and training curriculum.
- Evaluation metrics for boundary accuracy and focal length estimation.

Feel free to explore the code further and experiment with Depth Pro on your own images!

# The End

---