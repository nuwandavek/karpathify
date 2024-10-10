## Lesson 1: Introduction to DepthPro Architecture

In this lesson, we will explore the main `DepthPro` class and understand how it integrates the encoder and decoder to perform monocular depth estimation.

**Note:** The `DepthPro` class is the main module that combines the encoder and decoder to produce depth estimates from input images. It initializes the encoder and decoder, sets up the final convolution layers (`self.head`), and optionally includes a field-of-view (FOV) estimation network.

This class is crucial as it defines the overall architecture of the DepthPro model, orchestrating how features are extracted and processed.

To learn more about PyTorch module structures and how models are built, you can read the [PyTorch documentation on `nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).

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
        """Initialize DepthPro.

        Args:
        ----
            encoder: The DepthProEncoder backbone.
            decoder: The MultiresConvDecoder decoder.
            last_dims: The dimension for the last convolution layers.
            use_fov_head: Whether to use the field-of-view head.
            fov_encoder: A separate encoder for the field of view.

        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
        dim_decoder = decoder.dim_decoder
        self.head = nn.Sequential(
            nn.Conv2d(
                dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        # Set the final convolution layer's bias to be 0.
        self.head[4].bias.data.fill_(0)

        # Set the FOV estimation head.
        if use_fov_head:
            self.fov = FOVNetwork(num_features=dim_decoder, fov_encoder=fov_encoder)
```

**Note:** The `forward` method defines how the input image `x` is processed through the DepthPro model. It passes the image through the encoder to get multi-resolution encodings, decodes these features using the decoder, and then applies the final convolution layers to get the depth prediction.

Optionally, it also estimates the field of view if the FOV head is included.

Understanding the `forward` method is key to seeing how data flows through the network.

For more on defining forward methods in PyTorch, see [Defining `forward` functions in PyTorch](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html).

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Decode by projection and fusion of multi-resolution encodings."""
    _, _, H, W = x.shape
    assert H == self.img_size and W == self.img_size

    encodings = self.encoder(x)
    features, features_0 = self.decoder(encodings)
    canonical_inverse_depth = self.head(features)

    fov_deg = None
    if hasattr(self, "fov"):
        fov_deg = self.fov.forward(x, features_0.detach())

    return canonical_inverse_depth, fov_deg
```

## Lesson 2: Understanding the DepthPro Encoder

In this lesson, we will dive into the `DepthProEncoder`, focusing on how it processes multi-scale images and overlapping patches to create multi-resolution encodings.

**Note:** The `DepthProEncoder` class is responsible for generating multi-resolution encodings by processing the input image at multiple scales. It uses a patch encoder to process patches of the image and an image encoder for capturing global context.

By understanding this class, we can see how the model captures both local and global features, which is essential for accurate depth estimation.

To learn more about encoder architectures and multi-scale processing, consider reading about [Convolutional Neural Network (CNN) architectures](https://cs231n.github.io/convolutional-networks/).

```python
class DepthProEncoder(nn.Module):
    """DepthPro Encoder.

    An encoder aimed at creating multi-resolution encodings from Vision Transformers.
    """

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: nn.Module,
        image_encoder: nn.Module,
        hook_block_ids: Iterable[int],
        decoder_features: int,
    ):
        """Initialize DepthProEncoder.

        Args:
        ----
            dims_encoder: Dimensions of the encoder at different layers.
            patch_encoder: Backbone used for patches.
            image_encoder: Backbone used for global image encoder.
            hook_block_ids: Hooks to obtain intermediate features.
            decoder_features: Number of feature outputs in the decoder.

        """
        super().__init__()

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        self.hook_block_ids = list(hook_block_ids)
        # Further initialization...
```

**Note:** The `_create_pyramid` method generates a three-level image pyramid by downsampling the input image to different scales. This allows the encoder to process the image at multiple resolutions, capturing features at different levels of detail.

Understanding image pyramids is important for grasping how multi-scale features are extracted.

For more on image pyramids, you can read about [Image Pyramids in Computer Vision](https://en.wikipedia.org/wiki/Pyramid_(image_processing)).

```python
def _create_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a 3-level image pyramid."""
        x0 = x
        x1 = F.interpolate(
            x, size=None, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        x2 = F.interpolate(
            x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False
        )
        return x0, x1, x2
```

**Note:** The `split` method divides the input image into overlapping patches using a sliding window approach. The `overlap_ratio` parameter controls how much the patches overlap. This is useful for capturing local features while maintaining continuity across patches.

Sliding window methods are common in computer vision for tasks like object detection.

To learn more about sliding window techniques, see [Sliding Window Object Detection](https://medium.com/@nikasa1889/sliding-window-object-detection-with-python-and-opencv-python-be6a4792f3c3).

```python
def split(self, x: torch.Tensor, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into small patches with a sliding window."""
        patch_size = 384
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)
```

## Lesson 3: Decoding Multi-Resolution Features

In this lesson, we will explore the `MultiresConvDecoder` and understand how it decodes and fuses multi-resolution features to produce the final depth map.

**Note:** The `MultiresConvDecoder` class takes the multi-resolution features from the encoder and decodes them to produce the depth map. It projects features from different resolutions to a common decoder dimension and fuses them using convolutional layers.

This decoder plays a crucial role in combining features from multiple scales into a comprehensive representation.

For more on decoder architectures in neural networks, you can read about [U-Net Architecture](https://arxiv.org/abs/1505.04597), which is popular in segmentation tasks.

```python
class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        dim_decoder: int,
    ):
        """Initialize multiresolution convolutional decoder.

        Args:
        ----
            dims_encoder: Expected dims at each level from the encoder.
            dim_decoder: Dim of decoder features.

        """
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.dim_decoder = dim_decoder
        self.dim_out = dim_decoder

        # Initialization of convolutions and fusions...
        # Further code...
```

**Note:** In the `forward` method, the decoder processes the list of encoder outputs from the lowest to the highest resolution. It projects each encoding to the decoder dimension using convolutions, then fuses them using `FeatureFusionBlock2d` modules. This sequential fusion helps in reconstructing detailed spatial information.

Understanding this fusion process is important for grasping how multi-scale features contribute to the final prediction.

For more on feature fusion techniques, consider reading [Feature Pyramid Networks (FPN) for Object Detection](https://arxiv.org/abs/1612.03144).

```python
def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        """Decode the multi-resolution encodings."""
        num_levels = len(encodings)

        # Project features to the decoder dimension and fuse them
        features = self.convs[-1](encodings[-1])
        features = self.fusions[-1](features)
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)
        return features, lowres_features
```

**Note:** The `FeatureFusionBlock2d` class is a key component in the decoder for merging features from different resolutions. It uses residual blocks and optional deconvolution (upsampling) to adjust feature maps before fusing them.

This module helps in preserving spatial details while combining features.

To learn more about residual connections and their benefits, you can read the paper on [Deep Residual Learning](https://arxiv.org/abs/1512.03385).

```python
class FeatureFusionBlock2d(nn.Module):
    """Feature fusion for DPT."""

    def __init__(
        self,
        num_features: int,
        deconv: bool = False,
        batch_norm: bool = False,
    ):
        """Initialize feature fusion block.

        Args:
        ----
            num_features: Input and output dimensions.
            deconv: Whether to use deconvolution before the final output conv.
            batch_norm: Whether to use batch normalization in resnet blocks.

        """
        super().__init__()

        # Initialization of residual blocks and convolution layers...
        # Further code...
```

## Lesson 4: Estimating Field of View with FOVNetwork

In this lesson, we will learn how DepthPro estimates the field of view using the `FOVNetwork` and why this is important for depth estimation.

**Note:** The `FOVNetwork` class estimates the field of view (FOV) from the input features. Knowing the FOV is crucial for producing metric depth maps with absolute scale, especially when camera intrinsics are not available.

By estimating the FOV directly from the image, the model can adjust depth predictions to be more accurate in real-world units.

For more on the importance of FOV in depth estimation, consider reading about [Camera Models and Projection](https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_Models_Projection.pdf).

```python
class FOVNetwork(nn.Module):
    """Field of View estimation network."""

    def __init__(
        self,
        num_features: int,
        fov_encoder: Optional[nn.Module] = None,
    ):
        """Initialize the Field of View estimation block.

        Args:
        ----
            num_features: Number of features used.
            fov_encoder: Optional encoder to bring additional network capacity.

        """
        super().__init__()

        # Define FOV estimation head...
        # Further code...
```

**Note:** The `forward` method processes the input image and low-resolution features to estimate the FOV. If an additional encoder (`fov_encoder`) is provided, it extracts features from the downsampled image and combines them with the existing features.

This method shows how the model can predict camera parameters directly from the input data.

For further reading on estimating camera intrinsics from images, see [Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).

```python
def forward(self, x: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
        """Forward the FOV network.

        Args:
        ----
            x (torch.Tensor): Input image.
            lowres_feature (torch.Tensor): Low resolution feature.

        Returns:
        -------
            The field of view tensor.

        """
        if hasattr(self, "encoder"):
            x = F.interpolate(
                x,
                size=None,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            )
            x = self.encoder(x)[:, 1:].permute(0, 2, 1)
            lowres_feature = self.downsample(lowres_feature)
            x = x.reshape_as(lowres_feature) + lowres_feature
        else:
            x = lowres_feature
        return self.head(x)
```

## Lesson 5: Integrating Vision Transformers into DepthPro

In this final lesson, we will understand how Vision Transformers (ViTs) are integrated into the DepthPro architecture, focusing on loading and adapting pre-trained ViT models.

**Note:** The `create_vit` function loads a Vision Transformer model using a preset configuration. It can load pre-trained weights and adjust the model settings, such as image size and patch size, to fit the DepthPro architecture.

Integrating ViTs allows the model to leverage powerful transformer-based feature extraction.

To learn more about Vision Transformers, you can read the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

```python
def create_vit(
        preset: ViTPreset,
        use_pretrained: bool = False,
        checkpoint_uri: str | None = None,
        use_grad_checkpointing: bool = False,
    ) -> nn.Module:
        """Create and load a VIT backbone module.

        Args:
        ----
            preset: The VIT preset to load the pre-defined config.
            use_pretrained: Load pretrained weights if True.
            checkpoint_uri: Checkpoint to load the weights from.
            use_grad_checkpointing: Use gradient checkpointing.

        Returns:
        -------
            A Torch ViT backbone module.

        """
        config = VIT_CONFIG_DICT[preset]

        img_size = (config.img_size, config.img_size)
        patch_size = (config.patch_size, config.patch_size)

        model = timm.create_model(
            config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
        )
        # Adjust model as needed...
        # Further code...
```

**Note:** The `resize_vit` function adjusts the Vision Transformer to work with images of a different size by resizing its positional embeddings. This is necessary because ViTs use fixed-size position embeddings, and when the input image size changes, the embeddings need to be adapted.

Understanding how to adjust positional embeddings is important for customizing ViTs for various tasks.

For more on positional embeddings in transformers, see [Sinusoidal Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/).

```python
def resize_vit(model: nn.Module, img_size) -> nn.Module:
    """Resample the ViT module to the given size."""
    patch_size = model.patch_embed.patch_size
    model.patch_embed.img_size = img_size
    grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
    model.patch_embed.grid_size = grid_size

    pos_embed = resample_abs_pos_embed(
        model.pos_embed,
        grid_size,
        num_prefix_tokens=(
            0 if getattr(model, "no_embed_class", False) else model.num_prefix_tokens
        ),
    )
    model.pos_embed = torch.nn.Parameter(pos_embed)

    return model
```

**Note:** The `ViTConfig` class and `VIT_CONFIG_DICT` dictionary define configurations for different ViT presets. These configurations specify parameters like input channels, embedding dimensions, and image sizes.

By organizing configurations, the model can easily switch between different ViT variants.

For more on model configuration management, you can read about [Using Config Files in PyTorch](https://pytorch.org/tutorials/intermediate/model_serialization_torchscript.html).

```python
# In vit.py
class ViTConfig:
    """Configuration for ViT."""

    in_chans: int
    embed_dim: int
    # Further configuration fields...

VIT_CONFIG_DICT: Dict[ViTPreset, ViTConfig] = {
    "dinov2l16_384": ViTConfig(
        in_chans=3,
        embed_dim=1024,
        # Further settings...
    ),
}

```


