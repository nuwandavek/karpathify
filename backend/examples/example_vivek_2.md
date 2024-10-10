## Lesson 1: Introduction to Monocular Depth Estimation and Vision Transformers

Understanding the basics of monocular depth estimation and the role of Vision Transformers (ViT) in computer vision tasks.

**Note:** ### Understanding Monocular Depth Estimation
Monocular depth estimation involves predicting depth information from a single image. It’s a challenging task because there is no explicit depth cue as in stereo images.

### Vision Transformers (ViT) in Computer Vision
ViTs apply the Transformer architecture, originally used in NLP, to image data by breaking images into patches and treating them as token sequences. This allows the model to capture long-range dependencies and global context.

In `DepthPro`, Vision Transformers are utilized to process image patches and extract features for depth estimation.

```python
# depth_pro.py

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
        # Initialization code...
        pass
```

## Lesson 2: Understanding the DepthPro Architecture

Exploring the high-level architecture of DepthPro, including its encoder, decoder, and head components.

**Note:** ### DepthPro Components
- **Encoder**: Processes the input image to extract multi-scale features.
- **Decoder**: Fuses the multi-scale features to produce a high-resolution depth map.
- **Head**: Produces the final depth prediction from decoder features.

The `DepthPro` class initializes these components, preparing the network for end-to-end depth estimation.

```python
# depth_pro.py

class DepthPro(nn.Module):
    # ...
    def __init__(
        self,
        encoder: DepthProEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        use_fov_head: bool = True,
        fov_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # ...
```

## Lesson 3: Multi-Scale Vision Transformer Encoder Implementation

Delving into the implementation of the multi-scale ViT encoder in DepthPro and how it processes images at multiple scales.

**Note:** ### Multi-Scale Processing
The encoder creates an image pyramid with images at different scales. At each scale, overlapping patches are generated and processed using the same `patch_encoder` (ViT).

- **Shared Weights**: Using the same encoder across scales helps the model learn scale-invariant features.
- **Overlapping Patches**: Overlaps prevent seams and ensure smooth feature representations.

### Forward Method
The `forward` method orchestrates the multi-scale encoding:
- Creates image pyramid.
- Splits images into patches.
- Processes patches with the `patch_encoder`.
- Merges encoded patches to form multi-resolution feature maps.

```python
# encoder.py

class DepthProEncoder(nn.Module):
    # ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode input at multiple resolutions."""
        # Create image pyramid
        x0, x1, x2 = self._create_pyramid(x)

        # Generate patches at each scale
        x0_patches = self.split(x0, overlap_ratio=0.25)
        x1_patches = self.split(x1, overlap_ratio=0.5)
        x2_patches = x2

        # Process patches with shared ViT encoder
        # ...
```

## Lesson 4: Decoding and Fusion for High-Resolution Depth Maps

Understanding how the decoder combines multi-scale features to produce high-resolution depth maps.

**Note:** ### Multi-Resolution Decoding
The `MultiresConvDecoder` takes the multi-scale features from the encoder and progressively fuses them. This involves:

- Projecting features from different scales to a common dimension.
- Upsampling lower-resolution features to match higher-resolution ones.
- Fusing features using convolutional layers to combine information.

### Feature Fusion
The fusion is typically performed using residual blocks and can involve techniques like skip connections to preserve spatial information.

```python
# decoder.py

class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        dim_decoder: int,
    ):
        # Initialization code...
        pass

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        """Decode the multi-resolution encodings."""
        # Feature fusion and upsampling
        # ...
```

**Note:** ### Combining Encoder and Decoder
In the `forward` method of `DepthPro`, the encoder processes the input image to extract features, which are then decoded to produce the depth map.

- **Encoder Output**: Multi-scale features.
- **Decoder Output**: Fused features at high resolution.
- **Head Output**: Final depth prediction.

```python
# depth_pro.py

class DepthPro(nn.Module):
    # ...
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Encoding
        encodings = self.encoder(x)

        # Decoding
        features, _ = self.decoder(encodings)

        # Final depth prediction
        canonical_inverse_depth = self.head(features)

        # ...
```

## Lesson 5: Focal Length Estimation and Training Protocols in DepthPro

Exploring focal length estimation within the network and understanding the two-stage training curriculum.

**Note:** ### Focal Length Estimation
The `FOVNetwork` predicts the field of view (FOV) from features extracted by the encoder. This allows DepthPro to produce metric depth maps without relying on known camera intrinsics.

- **Input**: Features from the encoder.
- **Output**: Estimated FOV, which is used to calculate focal length in pixels.

### Integration with DepthPro
In `DepthPro`, the FOV estimation is integrated into the forward pass if `use_fov_head` is `True`.

```python
# fov.py

class FOVNetwork(nn.Module):
    """Field of View estimation network."""

    def __init__(
        self,
        num_features: int,
        fov_encoder: Optional[nn.Module] = None,
    ):
        # Initialization code...
        pass

    def forward(self, x: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
        # FOV estimation logic
        # ...
```

**Note:** ### Two-Stage Training Curriculum
DepthPro uses a two-stage training process:

1. **Stage One**: Train on a mix of real and synthetic datasets to learn generalizable features.
2. **Stage Two**: Fine-tune on synthetic datasets to sharpen boundaries and capture fine details.

This curriculum helps the model generalize well to unseen data while maintaining high-resolution depth predictions.

### Forward Pass with FOV Estimation
In the `forward` method, after predicting the depth, the model optionally predicts the FOV, which is then used to adjust the depth predictions to have absolute scale.

```python
# depth_pro.py

class DepthPro(nn.Module):
    # ...
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # ...
        canonical_inverse_depth = self.head(features)

        fov_deg = None
        if hasattr(self, "fov"):
            fov_deg = self.fov.forward(x, features_0.detach())

        return canonical_inverse_depth, fov_deg
```

