## Lesson 1: Introduction to Vision Transformers in Depth Estimation

Learn how Vision Transformers (ViT) are utilized as backbone models in DepthPro for depth estimation.

**Note:** ## Understanding the ViT Backbone

In DepthPro, Vision Transformers (ViTs) are employed as the backbone for feature extraction, leveraging their ability to capture global context. The `make_vit_b16_backbone` function adapts a pre-trained ViT model to be used within the DepthPro architecture.

### Key Components:
- **Model adaptation**: The function wraps a pre-trained ViT model, setting up necessary attributes and methods for it to interface correctly with the DepthPro encoder.
- **Feature hooks**: The `encoder_feature_layer_ids` specify which layers of the ViT model will be used to extract features at different resolutions.

### Relevant Sections from the Paper:

As mentioned in **Section 3.1** of the paper:

> *"The key idea of our architecture is to apply plain ViT encoders on patches extracted at multiple scales and fuse the patch predictions into a single high-resolution dense prediction in an end-to-end trainable model."*

This approach allows DepthPro to benefit from pre-trained ViT models while efficiently handling high-resolution images for depth estimation.

### Additional Reading:
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.
- [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413) by Ranftl et al.

### Formula:

The ViT splits an image into patches and processes them similarly to tokens in NLP transformers. If the image has dimensions $H \times W$, and each patch has dimensions $P \times P$, then the number of patches $N$ is:

$$
N = \frac{H \times W}{P^2}
$$

These patches are linearly embedded and combined with positional embeddings before being fed into the transformer encoder.

```python
def make_vit_b16_backbone(
    model,
    encoder_feature_dims,
    encoder_feature_layer_ids,
    vit_features,
    start_index=1,
    use_grad_checkpointing=False,
) -> nn.Module:
    """Make a ViT-B16 backbone for the DPT model."""
    if use_grad_checkpointing:
        model.set_grad_checkpointing()

    vit_model = nn.Module()
    vit_model.hooks = encoder_feature_layer_ids
    vit_model.model = model
    vit_model.features = encoder_feature_dims
    vit_model.vit_features = vit_features
    vit_model.model.start_index = start_index
    vit_model.model.patch_size = vit_model.model.patch_embed.patch_size
    vit_model.model.is_vit = True
    vit_model.model.forward = vit_model.model.forward_features

    return vit_model
```

## Lesson 2: Building the DepthPro Encoder

Understand how the DepthProEncoder processes multi-scale inputs and produces multi-resolution encodings.

**Note:** ## DepthPro Encoder

The `DepthProEncoder` is responsible for creating multi-resolution encodings from the input image by processing it at multiple scales.

### Key Components:
- **Multi-scale Processing**: The encoder downsamples the input image to create an image pyramid and processes patches at different scales.
- **Patch and Image Encoders**: Combines outputs from a patch-based encoder (processing image patches) and an image encoder (processing the whole image at a lower resolution).
- **Feature Fusion**: The outputs are merged to produce multi-resolution features for the decoder.

### Relevant Sections from the Paper:

From **Section 3.1**:

> *"The whole network operates at a fixed resolution of 1536 × 1536...At each scale, the patches are fed into the patch encoder, which produces a feature tensor at resolution 24 × 24 per input patch...We merge the feature patches into maps, which are fed into the decoder module."*

### Image Pyramid Creation

An image pyramid is created by downsampling the input image:

$$
I_0 = I \\
I_1 = \text{Downsample}(I, \text{scale}=0.5) \\
I_2 = \text{Downsample}(I, \text{scale}=0.25)
$$

### Additional Reading:
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) by Lin et al.
- [Understanding Multi-Scale Feature Hierarchies](https://distill.pub/2019/computing-receptive-fields/)

### Code Walkthrough:

- The `__init__` method initializes the encoder with the given encoders and configurations.
- The `forward` method processes the input image through the pyramid, splits it into patches, encodes them, and merges the features.

```python
class DepthProEncoder(nn.Module):
    """DepthPro Encoder combining patch and image encoders."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: nn.Module,
        image_encoder: nn.Module,
        hook_block_ids: Iterable[int],
        decoder_features: int,
    ):
        """Initialize DepthProEncoder."""
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        self.hook_block_ids = list(hook_block_ids)

        # Additional initialization code...

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode input at multiple resolutions."""
        # Implementation of the forward pass...

        return [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]
```

## Lesson 3: Designing the DepthPro Decoder

Explore how the decoder reconstructs depth maps from multi-resolution features using feature fusion and upsampling.

**Note:** ## Multiresolution Convolutional Decoder

The `MultiresConvDecoder` takes the multi-resolution encodings from the encoder and reconstructs the depth map through a series of convolutional and upsampling layers.

### Key Components:
- **Feature Fusion Blocks**: Uses `FeatureFusionBlock2d` to combine features from different scales.
- **Upsampling**: Gradually increases the spatial resolution of the feature maps to reconstruct the high-resolution depth map.
- **Residual Connections**: Implements `ResidualBlock` to facilitate better gradient flow and learning.

### Relevant Sections from the Paper:

From **Section 3.1**:

> *"Features are merged into a single high-resolution output through a decoder module, which resembles the DPT decoder."*

The decoder is designed to effectively combine multi-scale features and reconstruct detailed depth maps.

### Residual Blocks

As introduced by He et al., residual blocks help in training deeper networks by allowing gradients to flow directly through skip connections.

### Formula:

Residual connection:

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \mathcal{W}) + \mathbf{x}
$$

where:

- $\mathbf{x}$ is the input to the block.
- $\mathcal{F}(\mathbf{x}, \mathcal{W})$ is the residual function (e.g., convolution, batch norm, and ReLU).

### Additional Reading:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by He et al.
- [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413) by Ranftl et al.

### Code Walkthrough:

- The `__init__` method sets up convolutional layers and fusion blocks for combining features.
- The `forward` method processes the encoder outputs, fusing and upsampling them to produce the final feature map.

```python
class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        dim_decoder: int,
    ):
        """Initialize multiresolution convolutional decoder."""
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.dim_decoder = dim_decoder

        # Additional initialization code...

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        """Decode the multi-resolution encodings."""
        # Implementation of the forward pass...

        return features, lowres_features
```

## Lesson 4: Implementing Field of View Estimation

Learn how DepthPro estimates the field of view (focal length) from an input image to produce metric depth maps.

**Note:** ## Field of View Estimation in DepthPro

DepthPro estimates the field of view (FoV) or focal length directly from the input image, enabling metric depth estimation without requiring camera intrinsics.

### Key Components:
- **FOV Network**: A neural network that predicts the FoV using features from the encoder.
- **Integration with DepthPro**: The FoV estimation is integrated into the depth estimation pipeline, allowing for depth scaling.

### Relevant Sections from the Paper:

From **Section 3.3**:

> *"To handle images that may have inaccurate or missing EXIF metadata, we supplement our network with a focal length estimation head."*

The network predicts the horizontal angular field-of-view ($\theta$), which is then used to compute the focal length $f_{px}$ in pixels:

$$
f_{px} = \frac{0.5 \times w}{\tan(0.5 \times \theta)}
$$

where $w$ is the image width.

### Importance of FoV Estimation

Accurate FoV estimation allows DepthPro to produce metric depth maps with absolute scale, even when camera intrinsics are unavailable.

### Additional Reading:
- [Towards Zero-Shot Scale-Aware Monocular Depth Estimation](https://arxiv.org/abs/2304.08484) by Guizilini et al.
- [Learning the Camera Intrinsics of Streaming Videos](https://arxiv.org/abs/2104.14567) by Kocabas et al.

### Code Walkthrough:

- The `FOVNetwork` uses convolutional layers to regress the FoV from image features.
- The `forward` method processes the input features and returns the estimated FoV.

```python
class FOVNetwork(nn.Module):
    """Field of View estimation network."""

    def __init__(
        self,
        num_features: int,
        fov_encoder: Optional[nn.Module] = None,
    ):
        """Initialize the Field of View estimation block."""
        super().__init__()

        # Implementation details...

    def forward(self, x: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
        """Forward the fov network."""
        # Implementation of the forward pass...

        return self.head(x)
```

## Lesson 5: End-to-End Depth Estimation with DepthPro

Bring together all components to understand how DepthPro performs end-to-end depth estimation from an input image.

**Note:** ## DepthPro: End-to-End Depth Estimation

The `DepthPro` class combines the encoder, decoder, and FoV estimation to perform depth estimation from an input image.

### Key Components:
- **Encoder and Decoder**: Work together to extract features and reconstruct the depth map.
- **FoV Estimation**: If enabled, estimates the field of view for metric depth scaling.
- **Inference Method**: The `infer` method handles preprocessing and postprocessing to produce the final depth map.

### Relevant Sections from the Paper:

From **Section 3.2**:

> *"The network operates at a fixed resolution of 1536 × 1536... The outputs are merged to produce multi-resolution features for the decoder."*

And from **Section 3.3**:

> *"We supplement our network with a focal length estimation head... predicted canonical inverse depth is then scaled by the horizontal field of view."*

### Depth Scaling Equation

The predicted depth is scaled using the estimated focal length $f_{px}$ and the image width $w$:

$$
D_m = \frac{f_{px}}{w \times C}
$$

where $C$ is the predicted canonical inverse depth.

### Additional Reading:
- [Monocular Depth Estimation: An Overview](https://arxiv.org/abs/2006.05914)

### Code Walkthrough:

- The `DepthPro` class initializes with the encoder, decoder, and optionally the FoV network.
- The `forward` method computes the canonical inverse depth and optionally the FoV in degrees.
- The `infer` method handles the scaling of the canonical inverse depth to produce metric depth, considering the estimated or provided focal length.
- The method also resizes the depth map to the original image size if necessary.

```python
class DepthPro(nn.Module):
    """DepthPro network."""

    def __init__(
        self,
        encoder: DepthProEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        use_fov_head: bool = True,
        fov_encoder: Optional[nn.Module] = None,
    ):
        """Initialize DepthPro."""
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        
        # Additional initialization code...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode by projection and fusion of multi-resolution encodings."""
        # Implementation of the forward pass...

        return canonical_inverse_depth, fov_deg

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        f_px: Optional[Union[float, torch.Tensor]] = None,
        interpolation_mode="bilinear",
    ) -> Mapping[str, torch.Tensor]:
        """Infer depth and fov for a given image."""
        # Implementation of the inference method...

        return {
            "depth": depth.squeeze(),
            "focallength_px": f_px,
        }
```

**Note:** ## Running Inference with DepthPro

This sample code demonstrates how to use the `DepthPro` model to perform depth estimation on an input image.

### Steps:
1. **Load the Image**: Use PIL or another library to load the input image.
2. **Preprocess**: Apply the same transforms used during training to ensure consistency.
3. **Inference**: Set the model to evaluation mode and perform inference within `torch.no_grad()` to disable gradient calculations.
4. **Retrieve Output**: The output dictionary contains the depth map and estimated focal length (if applicable).

### Note on Preprocessing

The preprocessing transforms should match those used during training, which may include resizing, normalization, and conversion to tensor.

### Additional Notes

- Ensure that the input image size and aspect ratio are compatible with the model's expected input.
- The depth map produced is a single-channel tensor representing depth values in meters.

### Displaying the Depth Map

To visualize the depth map, you can convert it to a NumPy array and use matplotlib:

```python
import matplotlib.pyplot as plt

plt.imshow(depth_map.cpu().numpy(), cmap='plasma')
plt.colorbar(label='Depth in meters')
plt.show()
```

### Relevant Sections from the Paper:

From **Section 4** (Experiments):

> *"We release code and weights at https://github.com/apple/ml-depth-pro"*

This indicates that the model can be used as provided for inference tasks.

### Additional Reading:
- [PyTorch Documentation on Inference and Model Evaluation](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

```python
# Sample usage of DepthPro for inference

# Assuming `model` is an instance of DepthPro and `transform` is the input preprocessing function

from PIL import Image
import torch

# Load and preprocess the image
image = Image.open('path_to_image.jpg')
x = transform(image)  # Apply the same transforms used during training

# Perform inference
model.eval()
with torch.no_grad():
    output = model.infer(x)

# Retrieve the depth map
depth_map = output['depth']

# Display or save the depth map as needed
```


