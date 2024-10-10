## Lesson 1: Introduction to Monocular Depth Estimation and Depth Pro

In this lesson, you will learn the fundamentals of monocular depth estimation and get an overview of the Depth Pro model's capabilities.

**Note:** # Initialization of Depth Pro Model

To begin, we import the necessary modules and initialize the Depth Pro model along with the required transformations.

- **Lines 1-2**: Import the `create_model_and_transforms` function from the Depth Pro package and `torch`.
- **Line 5**: Set up the computation device (GPU if available, else CPU).
- **Lines 8-10**: Use the `create_model_and_transforms` function to instantiate the model and the image transformation pipeline. The function sets up the Depth Pro model with predefined configurations as described in the paper's Section 3.
- **Line 13**: Set the model to evaluation mode with `model.eval()`.

This setup allows us to use the Depth Pro model for inference on input images.


```python
from depth_pro import create_model_and_transforms
import torch

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model and transformation pipeline
model, transform = create_model_and_transforms(device=device)

# Switch model to evaluation mode
model.eval()
```

**Note:** # Preparing Input Image

Here, we load and preprocess an input image for the Depth Pro model.

- **Line 1**: Import the `Image` module from PIL for image handling.
- **Line 4**: Load your image and ensure it's in RGB format.
- **Line 7**: Apply the transformation pipeline to the image. The transformations include resizing, normalization, and conversion to a tensor suitable for the model. We also add a batch dimension with `unsqueeze(0)` and move the tensor to the computation device.

This prepares the image for depth estimation using the Depth Pro model.


```python
from PIL import Image

# Load an image
image = Image.open('your_image.jpg').convert('RGB')

# Apply transformations
input_tensor = transform(image).unsqueeze(0).to(device)
```

**Note:** # Performing Depth Estimation

We use the Depth Pro model to estimate the depth map of the input image.

- **Line 2**: Use a `torch.no_grad()` context to disable gradient calculations during inference.
- **Line 3**: Call the `infer` method of the model with the input tensor.
- **Line 6**: Extract the depth map from the output dictionary and convert it to a NumPy array for further processing or visualization.

At this point, `depth_map` contains the estimated depth information for the input image.

According to Section 1 of the paper, Depth Pro produces high-resolution depth maps with sharp boundaries and fine details, making it suitable for various applications such as novel view synthesis and image editing.


```python
# Perform inference
with torch.no_grad():
    output = model.infer(input_tensor)

# Extract depth map
depth_map = output['depth'].cpu().squeeze().numpy()
```

**Note:** # Visualizing the Estimated Depth Map

Finally, we visualize the depth map using Matplotlib.

- **Line 1**: Import Matplotlib for plotting.
- **Line 4**: Display the depth map using a colormap that highlights depth variations.
- **Lines 5-7**: Add a title, remove axes, and display the plot.

This visualization helps us appreciate the quality of the depth estimation produced by Depth Pro, showcasing its ability to capture fine-grained details as highlighted in the paper's Figure 1.


```python
import matplotlib.pyplot as plt

# Visualize the depth map
plt.imshow(depth_map, cmap='inferno')
plt.title('Estimated Depth Map')
plt.axis('off')
plt.show()
```

## Lesson 2: Multi-Scale Vision Transformer Architecture

In this lesson, you will explore the multi-scale Vision Transformer (ViT) architecture used in Depth Pro and understand how it captures both global context and fine details.

**Note:** # DepthPro Encoder Initialization

The `DepthProEncoder` class is responsible for encoding input images at multiple scales using Vision Transformers, as described in Section 3.1 of the paper.

- **Line 4**: The constructor (`__init__`) initializes the encoder with parameters such as the dimensions of encoder features, the patch and image encoders (both ViTs), the block IDs to hook for intermediate outputs, and the number of decoder features.
- **Lines 5-9**: The superclass is initialized, and instance variables are set.

The multi-scale approach allows the model to process patches at different resolutions, capturing both global context and fine details.

$$
\text{See Section 3.1: "The key idea of our architecture is to apply plain ViT encoders on patches extracted at multiple scales and fuse the patch predictions into a single high-resolution dense prediction in an end-to-end trainable model."}
$$


```python
# In depth_pro/network/encoder.py

class DepthProEncoder(nn.Module):
    def __init__(self, dims_encoder, patch_encoder, image_encoder, hook_block_ids, decoder_features):
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        self.hook_block_ids = list(hook_block_ids)
        # ... (initialization continues)
```

**Note:** # Creating an Image Pyramid

The `_create_pyramid` method generates a 3-level image pyramid from the input image.

- **Line 1**: Define the method `_create_pyramid`.
- **Line 2**: `x0` is the original input image.
- **Lines 3-4**: Downsample the image to half and quarter scales to obtain `x1` and `x2`.

This pyramid enables the model to process images at multiple scales, which is crucial for capturing details at different resolutions.

$$
\text{Refer to Section 3.1: "After downsampling to 1536×1536, the input image is split into patches of 384×384. For the two finest scales, we let patches overlap to avoid seams."}
$$


```python
def _create_pyramid(self, x):
    x0 = x  # Original resolution
    x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    x2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
    return x0, x1, x2
```

**Note:** # Splitting the Image into Overlapping Patches

The `split` method divides the image into overlapping patches.

- **Line 1**: Define the `split` method with an `overlap_ratio` parameter.
- **Lines 2-3**: Calculate the patch size and stride based on the overlap ratio.
- **Lines 4-5**: Determine the number of steps needed to cover the image.
- **Lines 7-12**: Loop over the image to extract patches and append them to a list.
- **Line 13**: Concatenate all patches into a single tensor.

Overlapping patches help the model avoid boundary artifacts and ensure seamless predictions across patches.

$$
\text{From Section 3.1: "At each scale, the patches are fed into the patch encoder, which produces a feature tensor... We merge the feature patches into maps."}
$$


```python
def split(self, x, overlap_ratio=0.25):
    patch_size = 384
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = x.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    x_patch_list = []
    for j in range(steps):
        for i in range(steps):
            i0 = i * patch_stride
            j0 = j * patch_stride
            patch = x[..., j0:j0+patch_size, i0:i0+patch_size]
            x_patch_list.append(patch)
    return torch.cat(x_patch_list, dim=0)
```

**Note:** # Forward Pass of the Encoder

The `forward` method executes the encoding process.

- **Line 1**: Define the `forward` method.
- **Line 2**: Create the image pyramid.
- **Lines 3-5**: Split images at different scales into patches.
- **Line 7**: Combine all patches into one tensor.
- **Line 8**: Process all patches through the shared patch encoder (ViT).

This step applies the Vision Transformer to patches from multiple scales, enabling the model to learn scale-invariant representations.

$$
\text{According to Section 3.1: "Intuitively, this may allow learning representations that are scale-invariant as weights are shared across scales."}
$$


```python
def forward(self, x):
    x0, x1, x2 = self._create_pyramid(x)
    x0_patches = self.split(x0, overlap_ratio=0.25)
    x1_patches = self.split(x1, overlap_ratio=0.5)
    x2_patches = x2  # Low-resolution image
    # Process patches with shared ViT encoder
    x_pyramid_patches = torch.cat((x0_patches, x1_patches, x2_patches), dim=0)
    x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
    # ... (forward pass continues)
```

## Lesson 3: Training Protocol and Loss Functions

In this lesson, you will understand the training strategy of Depth Pro, focusing on the loss functions that promote sharp depth estimates.

**Note:** # Forward Method in Depth Pro Model

The `forward` method computes the canonical inverse depth.

- **Line 3**: After encoding and decoding, pass the features through the `self.head` to obtain `canonical_inverse_depth`.
- **Line 4**: Return the canonical inverse depth map.

This output is used in the loss functions during training to compute errors with respect to ground truth.

$$
C = f(I) \quad \text{(Equation from Section 3.2)}
$$
where $C$ is the canonical inverse depth, $f$ is the model, and $I$ is the input image.


```python
# In depth_pro/depth_pro.py

def forward(self, x):
    # ... (encoder and decoder forward pass)
    canonical_inverse_depth = self.head(features)
    return canonical_inverse_depth
```

**Note:** # Loss Functions Implementation

The training uses several loss functions to promote sharp and accurate depth predictions.

- **Line 4**: Compute the Mean Absolute Error (MAE) between predicted and ground-truth canonical inverse depths.
  $$
  \mathcal{L}_{\mathit{MAE}}(\hat{C}, C) = \frac{1}{N} \sum_{i=1}^N | \hat{C}_i - C_i |
  $$
  (Refer to Equation 1 in Section 3.2)

- **Lines 7-14**: Define a multi-scale derivative loss function.
  - **Lines 9-12**: At each scale, downsample the predicted and ground-truth depths and compute the gradient difference.
  - **Line 13**: Accumulate the loss across scales.
  
These loss functions help the model learn fine details and sharp transitions in depth.

$$
\text{Equation 2 in Section 3.2 defines the multi-scale derivative loss:}
\mathcal{L}_{*, p, M}(C, \hat{C}) = \frac{1}{M} \sum_{j=1}^M \frac{1}{N_j} \sum_{i}^{N_j} | \nabla_* C_i^j - \nabla_* \hat{C}_i^j |^p
$$
where $\nabla_*$ represents spatial derivative operators.


```python
# Loss functions in training

L_MAE = torch.mean(torch.abs(C_pred - C_gt))

# Multi-scale derivative loss
def multi_scale_derivative_loss(C_pred, C_gt, scales=[1,2,4,8]):
    loss = 0
    for scale in scales:
        C_pred_scaled = F.avg_pool2d(C_pred, kernel_size=scale)
        C_gt_scaled = F.avg_pool2d(C_gt, kernel_size=scale)
        loss += torch.mean(torch.abs(gradient(C_pred_scaled) - gradient(C_gt_scaled)))
    return loss
```

**Note:** # Combining Loss Functions in Training

- **Line 3**: Combine the MAE loss with the multi-scale derivative loss weighted by a factor (e.g., 0.1).
- **Line 4**: Perform backpropagation with `loss.backward()`.

This demonstrates how multiple loss terms contribute to the overall training objective, encouraging both global accuracy and local detail preservation.

Refer to the training curriculum described in Section 3.2:

$$
\text{"In the first stage, we aim to learn robust features... In the second stage of training, designed to sharpen boundaries..."}
$$


```python
# Usage of loss functions during training

loss = L_MAE + 0.1 * multi_scale_derivative_loss(C_pred, C_gt)
loss.backward()
```

## Lesson 4: Evaluating Boundary Accuracy in Depth Maps

In this lesson, you will learn about the evaluation metrics for boundary accuracy in depth maps and how they are implemented using code.

**Note:** # Scale-Invariant Boundary F1 Score

The `SI_boundary_F1` function computes the boundary F1 score across multiple thresholds.

- **Line 3**: Define thresholds between `t_min` and `t_max`.
- **Lines 5-7**: For each threshold `t`, compute the boundary F1 score and collect the results.
- **Line 8**: Return the average F1 score across thresholds.

This metric assesses the alignment of predicted depth boundaries with the ground truth, emphasizing sharpness and accuracy at object edges.

Refer to Section 3.2 of the paper:

$$
\text{"We first define the metrics for depth maps... We then define an occluding contour..."}
$$


```python
# In depth_pro/eval/boundary_metrics.py

def SI_boundary_F1(predicted_depth, target_depth, t_min=1.05, t_max=1.25, N=10):
    thresholds = np.linspace(t_min, t_max, N)
    f1_scores = []
    for t in thresholds:
        f1 = boundary_f1(predicted_depth, target_depth, t)
        f1_scores.append(f1)
    return np.mean(f1_scores)
```

**Note:** # Computing Boundary F1 Score

The `boundary_f1` function calculates the F1 score for a given threshold.

- **Line 2**: Compute foreground-background relations for predicted (`pr`) and ground-truth (`gt`) depths.
- **Lines 4-7**: Calculate precision (`p`) and recall (`r`) based on overlaps of foreground-background relations.
- **Lines 9-11**: Compute the F1 score, handling the case where both `p` and `r` are zero.

This function is critical for evaluating how well the model captures depth discontinuities at object boundaries.

From Section 3.2:

$$
\text{"To evaluate boundary tracing in predicted depth maps, we use the pairwise depth ratio of neighboring pixels to define a foreground/background relationship."}
$$


```python
def boundary_f1(pr, gt, t):
    # Compute foreground-background relations
    ap, bp, cp, dp = fgbg_depth(pr, t)
    ag, bg, cg, dg = fgbg_depth(gt, t)
    # Compute precision and recall
    p = (np.sum(ap & ag) + np.sum(bp & bg) + np.sum(cp & cg) + np.sum(dp & dg)) / \
        (np.sum(ap) + np.sum(bp) + np.sum(cp) + np.sum(dp))
    r = (np.sum(ap & ag) + np.sum(bp & bg) + np.sum(cp & cg) + np.sum(dp & dg)) / \
        (np.sum(ag) + np.sum(bg) + np.sum(cg) + np.sum(dg))
    # Compute F1 score
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)
```

**Note:** # Applying the Boundary Metrics

- **Lines 2-3**: Assume `predicted_depth` is the output from the model and `target_depth` is the ground truth.
- **Line 4**: Compute the scale-invariant boundary F1 score.

This allows us to quantitatively assess the model's performance on preserving sharp boundaries in depth estimation.

As highlighted in Section 4:

$$
\text{"We find that Depth Pro produces more accurate boundaries than all baselines on all datasets, by a significant margin."}
$$


```python
# Example usage
predicted_depth = ...  # Output from model
target_depth = ...     # Ground truth depth
f1_score = SI_boundary_F1(predicted_depth, target_depth)
```

## Lesson 5: Zero-Shot Focal Length Estimation

In the final lesson, you will understand how Depth Pro performs zero-shot focal length estimation from a single image and explore its implementation.

**Note:** # Field of View Estimation Network

The `FOVNetwork` estimates the field of view (FoV) from image features.

- **Line 4**: Constructor initializes the network.
- **Lines 7-9**: Define the `self.head` as a sequence of convolutional layers ending with a single output channel representing the FoV.

Estimating the FoV allows the model to compute metric depth without camera intrinsics.

From Section 3.3:

$$
\text{"We supplement our network with a focal length estimation head... to predict the horizontal angular field-of-view."}
$$


```python
# In depth_pro/network/fov.py

class FOVNetwork(nn.Module):
    def __init__(self, num_features, fov_encoder=None):
        super().__init__()
        # ... (other initializations)
        self.head = nn.Sequential(
            nn.Conv2d(num_features // 8, 1, kernel_size=6, stride=1, padding=0),
        )
```

**Note:** # Forward Method for FOV Estimation

- **Line 1**: Define the `forward` method, taking image `x` and `lowres_feature` as inputs.
- **Line 2**: Use the low-resolution feature map extracted from the encoder.
- **Line 3**: Pass the features through the `self.head` to estimate the FoV.
- **Line 4**: Return the estimated FoV.

This output is used to compute the focal length in pixels:

$$
\text{From Section 3.3:}
D_m = \frac{f_{px}}{w C}
$$
where $D_m$ is the metric depth, $f_{px}$ is the focal length in pixels, $w$ is the image width, and $C$ is the canonical inverse depth.


```python
def forward(self, x, lowres_feature):
    x = lowres_feature
    fov = self.head(x)
    return fov
```

**Note:** # Computing Metric Depth Without Known Intrinsics

- **Line 3**: If `f_px` (focal length in pixels) is not provided, compute it from the estimated FoV.
  - **Line 4**: Calculate `f_px` using the formula $f_{px} = \frac{W}{2 \tan(\frac{\text{FoV}}{2})}$.
- **Line 5**: Adjust the inverse depth using the computed `f_px`.
- **Line 6**: Obtain the metric depth by inverting the adjusted inverse depth.
- **Line 8**: Return the depth map and focal length.

This demonstrates how Depth Pro estimates metric depth without relying on camera metadata, fulfilling the goal stated in the paper.

From Section 3.3:

$$
\text{"If the focal length is given, it is used to estimate the final metric depth, otherwise the model estimates $f_{px}$ to compute the depth metricness."}
$$


```python
# In depth_pro/depth_pro.py

def infer(self, x, f_px=None):
    # ... (forward pass to get canonical_inverse_depth and fov_deg)
    if f_px is None:
        f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg))
    inverse_depth = canonical_inverse_depth * (W / f_px)
    depth = 1.0 / inverse_depth
    return {'depth': depth, 'focallength_px': f_px}
```

