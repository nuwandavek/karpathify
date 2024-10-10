## Lesson 1: Introduction to Vision Transformers in DepthPro

In this lesson, we will explore the Vision Transformer (ViT) and how it is utilized in the DepthPro model as a backbone for feature extraction. We will examine how ViT models are created and integrated into the DepthPro architecture by studying the `vit_factory.py` and `vit.py` files.

**Note:** ### Creating a ViT Backbone

The `create_vit` function in `vit_factory.py` is responsible for creating and loading a Vision Transformer (ViT) model based on a given preset configuration. In the context of DepthPro, ViT serves as the backbone for feature extraction.

This function utilizes the `timm` library to create a ViT model:

```python
model = timm.create_model(
    config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
)
```

**Key Concepts:**
- **Vision Transformers (ViT):** Introduced in [Dosovitskiy et al., 2021], ViT applies the transformer architecture to images by splitting them into patches and processing them similarly to sequences in NLP tasks.
- **Pretrained Models:** Using pretrained models helps in leveraging learned features from large datasets, which is beneficial for tasks like depth estimation.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Discusses the use of ViT as the backbone in the DepthPro architecture.

**Further Reading:**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

```python
def create_vit(
    preset: ViTPreset,
    use_pretrained: bool = False,
    checkpoint_uri: str | None = None,
    use_grad_checkpointing: bool = False,
) -> nn.Module:
    """Create and load a VIT backbone module."""
    # ... Function implementation ...
```

**Note:** ### Integrating ViT into DepthPro

The `make_vit_b16_backbone` function in `vit.py` adapts a ViT model to be used as the encoder backbone in DepthPro. It configures the model to output features required by the decoder.

Key steps include:
- Setting up hooks to extract features from specific transformer blocks.
- Adjusting patch sizes and image sizes if necessary.

**Key Concepts:**
- **Feature Extraction:** Extracting features from intermediate layers helps in building multi-resolution representations.
- **Model Adaptation:** Modifying pretrained models to fit the requirements of a new architecture.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Details on how ViTs are integrated into the DepthPro encoder.

**Further Reading:**
- ViT adaptation techniques in [Ranftl et al., 2021]: Vision Transformers for Dense Prediction.

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
    # ... Function implementation ...
```

## Lesson 2: DepthPro Encoder and Multi-resolution Encoding

In this lesson, we will delve into the `DepthProEncoder` class in `encoder.py`. We will learn how the encoder creates multi-resolution encodings using a multi-scale image pyramid and a sliding window approach.

**Note:** ### Understanding the DepthProEncoder

The `DepthProEncoder` class is responsible for encoding the input image into multi-resolution feature maps. It achieves this by:

1. **Creating an Image Pyramid:** The input image is downsampled to multiple scales.
2. **Sliding Window Approach:** At each scale, the image is split into overlapping patches.
3. **Patch Encoding:** Each patch is passed through a shared ViT patch encoder.
4. **Feature Merging:** Encoded patches are merged back to form feature maps.

**Key Concepts:**
- **Multi-scale Processing:** Helps in capturing both global context and fine-grained details.
- **Shared Weights:** Using the same encoder for patches at different scales promotes scale-invariance.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Explains the multi-scale ViT-based architecture and the encoder's role.

**Further Reading:**
- Multi-scale feature extraction in CNNs and ViTs.

```python
class DepthProEncoder(nn.Module):
    """DepthPro Encoder."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: nn.Module,
        image_encoder: nn.Module,
        hook_block_ids: Iterable[int],
        decoder_features: int,
    ):
        # ... Constructor implementation ...
```

**Note:** ### Creating an Image Pyramid

The `_create_pyramid` method generates a three-level pyramid from the input image:

- **Level 0:** Original resolution.
- **Level 1:** Downsampled by a factor of 0.5.
- **Level 2:** Downsampled by a factor of 0.25.

This pyramid allows the model to process the image at multiple scales, capturing both local and global features.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Discusses the benefits of multi-scale processing in depth estimation.

```python
def _create_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a 3-level image pyramid."""
        # ... Function implementation ...
```

**Note:** ### Splitting into Overlapping Patches

The `split` method divides the image at each scale into overlapping patches using a sliding window. The overlap helps prevent boundary artifacts when merging the patches later.

**Key Concepts:**
- **Sliding Window:** A technique to process subsets of data with overlaps, useful in capturing context around patch borders.
- **Overlap Ratio:** Determines how much consecutive patches overlap with each other.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Mentions the use of overlapping patches to avoid seams.

```python
def split(self, x: torch.Tensor, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        # ... Function implementation ...
```

**Note:** ### Merging Encoded Patches

After encoding the patches, the `merge` method reconstructs the feature maps by stitching the patches back together, accounting for overlaps and removing padding.

**Key Concepts:**
- **Feature Stitching:** Combining features from patches to form a coherent feature map.
- **Padding Removal:** Ensures that overlapping regions are handled appropriately to avoid artifacts.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Describes how patch predictions are fused into a single high-resolution output.

```python
def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into an image with sliding window."""
        # ... Function implementation ...
```

## Lesson 3: Multi-resolution Convolutional Decoder in DepthPro

This lesson focuses on the `MultiresConvDecoder` class in `decoder.py`. We will understand how the decoder fuses features from different resolutions to produce the final depth map.

**Note:** ### Understanding the MultiresConvDecoder

The `MultiresConvDecoder` class decodes the multi-resolution features produced by the encoder to generate the depth map. Key components include:

- **Convolutions:** Project encoder features to a common dimension.
- **Feature Fusion Blocks:** Combine features from different scales progressively.

**Key Concepts:**
- **Feature Fusion:** Merging features from multiple resolutions to capture both coarse and fine details.
- **Progressive Upsampling:** Gradually increasing the spatial resolution of feature maps.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Discusses how features from different scales are fused in the decoder.

**Further Reading:**
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

```python
class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        dim_decoder: int,
    ):
        # ... Constructor implementation ...
```

**Note:** ### Feature Fusion Blocks

The `FeatureFusionBlock2d` class is used within the decoder to fuse feature maps from different scales. It includes:

- **Residual Blocks:** Enhance learning by allowing gradients to flow through skip connections.
- **Upsampling Layers:** To match the spatial dimensions of higher-resolution features.

**Key Concepts:**
- **Residual Learning:** Helps in training deep networks by mitigating the vanishing gradient problem.
- **Upsampling:** Increases the spatial dimensions of feature maps, essential for generating high-resolution outputs.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Mentions the use of feature fusion blocks in the decoder.

```python
class FeatureFusionBlock2d(nn.Module):
    """Feature fusion for DPT."""
    # ... Class implementation ...
```

## Lesson 4: Integrating the Encoder and Decoder: The DepthPro Network

In this lesson, we will look at how the encoder and decoder are integrated into the DepthPro network by examining the `depth_pro.py` file. We will explore how the model performs depth estimation and understand the inference process.

**Note:** ### The DepthPro Network

The `DepthPro` class integrates the encoder and decoder to form the full depth estimation model. Key components include:

- **Encoder and Decoder Integration:** The encoder extracts multi-resolution features, and the decoder fuses them to predict depth.
- **Depth Head:** A convolutional head that refines the decoder output to produce the final depth map.
- **Field of View (FoV) Estimation:** An optional module to estimate the camera's field of view.

**Key Concepts:**
- **End-to-End Architecture:** Combines all components into a single trainable model.
- **Depth Prediction Head:** Transforms decoder features into depth values.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Provides an overview of the entire DepthPro architecture.

**Further Reading:**
- Understanding end-to-end deep learning models.

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
        # ... Constructor implementation ...
```

**Note:** ### Forward Pass in DepthPro

The `forward` method defines how the input image is processed through the encoder and decoder to produce the depth map.

Steps:

1. **Encoding:** Pass the input through the `DepthProEncoder` to obtain multi-resolution features.
2. **Decoding:** Use the `MultiresConvDecoder` to fuse features and generate high-resolution representations.
3. **Depth Estimation:** Apply the depth head to produce the canonical inverse depth map.
4. **FoV Estimation:** Optionally estimate the field of view if the `use_fov_head` is enabled.

**Key Concepts:**
- **Canonical Inverse Depth:** A representation that prioritizes areas close to the camera.
- **Optional Modules:** Flexibility to include or exclude components like FoV estimation.

**Relevant Sections in the Paper:**
- *Section 3.1 Network*: Details the flow of data through the network.
- *Section 3.2 Sharp Monocular Depth Estimation*: Discusses the training objectives and loss functions.

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode by projection and fusion of multi-resolution encodings."""
        # ... Forward pass implementation ...
```

**Note:** ### Inference Method

The `infer` method is used for depth estimation during inference. It handles preprocessing, resizing, and post-processing to produce the final depth map.

Highlights:

- **Resizing Input:** Adjusts the input image to the network's expected size.
- **Handling Focal Length:** If the focal length is not provided, it estimates it using the FoV estimator.
- **Depth Calculation:** Converts the canonical inverse depth to metric depth.

**Key Concepts:**
- **Inference Efficiency:** Uses `@torch.no_grad()` to disable gradient calculations for faster inference.
- **Metric Depth Estimation:** Produces depth values with absolute scale.

**Relevant Sections in the Paper:**
- *Section 3.3 Focal Length Estimation*: Explains how the model estimates the focal length when it's not provided.

```python
@torch.no_grad()
def infer(
        self,
        x: torch.Tensor,
        f_px: Optional[Union[float, torch.Tensor]] = None,
        interpolation_mode="bilinear",
    ) -> Mapping[str, torch.Tensor]:
        """Infer depth and fov for a given image."""
        # ... Inference implementation ...
```

## Lesson 5: Running DepthPro and Field of View Estimation

In the final lesson, we will learn how to run the DepthPro model using the `run.py` script in the `cli` folder. We will also explore the `FOVNetwork` in `fov.py` to understand how the field of view is estimated.

**Note:** ### Running the DepthPro Model

The `run.py` script provides a command-line interface to run the DepthPro model on input images.

Key steps in `main()`:

- **Argument Parsing:** Uses `argparse` to handle command-line arguments such as input image path and output path.
- **Model Loading:** Calls `create_model_and_transforms()` to load the pre-trained DepthPro model.
- **Inference:** Processes the input image(s) and outputs the estimated depth map(s).

**How to Run:**

```bash
python run.py -i path/to/image.jpg -o path/to/output/
```

**Relevant Sections in the Paper:**
- *Section 4 Experiments*: Discusses the practical applications and testing of the model.

```python
# cli/run.py

def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    # ... Argument parsing and setup ...
    
    run(parser.parse_args())
```

**Note:** ### Field of View Estimation

The `FOVNetwork` class in `fov.py` is responsible for estimating the camera's field of view (FoV) from the input image.

Key components:

- **Feature Extraction:** Utilizes features from the low-resolution encoder output.
- **Convolutional Layers:** Processes features to predict the FoV.

**Purpose of FoV Estimation:**

- **Metric Depth Calculation:** Knowing the FoV allows the model to produce depth maps with absolute scale without requiring camera intrinsics.

**Relevant Sections in the Paper:**
- *Section 3.3 Focal Length Estimation*: Discusses the importance of estimating the focal length or FoV when metadata is unavailable.

**Further Reading:**
- [Camera Models and Perspective Projection](https://math.mit.edu/~djk/18.357/notes/perspective.pdf)

```python
class FOVNetwork(nn.Module):
    """Field of View estimation network."""

    def __init__(
        self,
        num_features: int,
        fov_encoder: Optional[nn.Module] = None,
    ):
        # ... Constructor implementation ...
```

**Note:** ### Inference with FoV Estimation

When running inference, if the focal length (`f_px`) is not provided, the model uses the `FOVNetwork` to estimate it.

**Usage Notes:**
- If you have the camera's focal length, pass it to `infer()` to improve accuracy.
- The estimated focal length can be retrieved from the prediction dictionary.

**Relevant Sections in the Paper:**
- *Section 3.3 Focal Length Estimation*: Explains the estimation process and its impact on depth accuracy.

```python
# Running the inference with FoV estimation

prediction = model.infer(transform(image), f_px=None)

# Extract the depth and estimated focal length
depth = prediction["depth"].detach().cpu().numpy().squeeze()
focallength_px = prediction["focallength_px"]
```


