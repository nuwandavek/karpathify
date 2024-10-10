## Lesson 1: Introduction to Vision Transformers in DepthPro

In this lesson, we will explore the Vision Transformer (ViT) and how it is utilized in the DepthPro model as a backbone for feature extraction. We will examine how ViT models are created and integrated into the DepthPro architecture by studying the `vit_factory.py` and `vit.py` files.

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

```python
def _create_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a 3-level image pyramid."""
        # ... Function implementation ...
```

```python
def split(self, x: torch.Tensor, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        # ... Function implementation ...
```

```python
def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into an image with sliding window."""
        # ... Function implementation ...
```

## Lesson 3: Multi-resolution Convolutional Decoder in DepthPro

This lesson focuses on the `MultiresConvDecoder` class in `decoder.py`. We will understand how the decoder fuses features from different resolutions to produce the final depth map.

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

```python
class FeatureFusionBlock2d(nn.Module):
    """Feature fusion for DPT."""
    # ... Class implementation ...
```

## Lesson 4: Integrating the Encoder and Decoder: The DepthPro Network

In this lesson, we will look at how the encoder and decoder are integrated into the DepthPro network by examining the `depth_pro.py` file. We will explore how the model performs depth estimation and understand the inference process.

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

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode by projection and fusion of multi-resolution encodings."""
        # ... Forward pass implementation ...
```

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

```python
# Running the inference with FoV estimation

prediction = model.infer(transform(image), f_px=None)

# Extract the depth and estimated focal length
depth = prediction["depth"].detach().cpu().numpy().squeeze()
focallength_px = prediction["focallength_px"]
```


